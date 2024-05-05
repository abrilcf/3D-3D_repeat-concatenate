import torch
import argparse
import os
import csv
import random
import yaml
import wandb
import sys
import copy
import monai
from yaml.loader import SafeLoader
from tabulate import tabulate

import torch.utils.data
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.backends import cudnn
from tqdm import tqdm

from src.optim import DecayLR, get_optimizer, freeze, unfreeze, calculate_adaptive_weight, update_ema, self_supervised_loss
from src.unet import UNet3D
from src.resnet2 import ResNet_D3D
from src.dataloader import XCT_dataset
from src.diffaug import DiffAugment
from src.utils import kaiming_weights_init, load_models_from_checkpoint, get_minibatch, get_views, eval_psnr_ssim, update_config, save_config, save_ct_slices

##########################################################################

parser = argparse.ArgumentParser(
    description="OT 2D-3D Translation")
parser.add_argument('--config', type=str, default=None,
                   help='Path to config file')
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--sample_noise", action="store_true", help="Concatenates noise instead of duplicating views")
parser.add_argument("--rough_align", action="store_true", help="Apply rought alignment to 2d views")
parser.add_argument('--model', type=str, default='swin-unetr',
                    help='Mapping network model [unet, swin-unetr]')


args, cfg_args = parser.parse_known_args()

with open(args.config) as config_file:
    config = yaml.load(config_file, Loader=SafeLoader)
    config = update_config(config, cfg_args)

t_val = []
for arg in vars(args):
    t_val.append([arg, getattr(args, arg)])
print('\n')
print(tabulate(t_val, 
               ['input', 'value'],
               tablefmt="psql"))

##########################################################################
def update_model_weights(optim, loss, amp=False, scaler=None):
    if amp:
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
    else:
        loss.backward()
        optim.step()

##########################################################################

dataset = config['data']['dataset']

results_dir = os.path.join(f"{config['training']['outf']}", f"{config['experiment']}")
weights_dir = os.path.join(results_dir, dataset, "weights")
test_dir = os.path.join(results_dir, dataset, "test")
test_file = os.path.join(test_dir, 'res_psnr_ssm.csv')

try:
    os.makedirs(results_dir, exist_ok=True)
except OSError:
    pass
try:
    os.makedirs(test_dir, exist_ok=True)
except OSError:
    pass
try:
    os.makedirs(weights_dir, exist_ok=True)
except OSError:
    pass

save_config(os.path.join(results_dir, dataset, 'config.yaml'), config)


with open(test_file, 'w', encoding='UTF8') as csv_stat:
    csv_writer = csv.writer(csv_stat)
    csv_writer.writerow(['epoch', 'PSNR', 'SSIM'])
    
if config['training']['manualSeed'] is None:
    manualSeed = random.randint(1, 10000)
else: manualSeed = config['training']['manualSeed']
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True
wandb.init(name=config['experiment'], project='ot_x2ct_3d_paired', config=config, mode=config['training']['wandb_mode'])

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Dataset ops
num_xrays = config['data']['num_proj']
datadir = config['data']['data_dir']
image_size = config['data']['image_size']
batch_size = config['data']['batch_size']

if dataset in ['chest', 'knee', 'midrc', 'anti-pd-1', 'covid-19-ar', 'lctsc', 'nsclc', 'spie-aapm']:
    train_dataset = XCT_dataset(data_dir=datadir, train=True, dataset=dataset,
                                xray_scale=image_size,
                                projections=num_xrays,
                                f16=config['training']['amp'],
                                separate_xrays=False,
                                use_synthetic=config['data']['synthetic'])
    valid_dataset = XCT_dataset(data_dir=datadir, train=False, dataset=dataset,
                                xray_scale=image_size,
                                projections=num_xrays,
                                f16=config['training']['amp'],
                                separate_xrays=False,
                                use_synthetic=config['data']['synthetic'])
else:
    raise Exception("Unknown option set for data:dataset")

dataloader = DataLoader(train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=config['data']['workers'],
                        pin_memory=True)
train_dataloader = iter(dataloader)

valid_loader = DataLoader(valid_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=0,
                          pin_memory=True)
valid_dataloader = iter(valid_loader)

try:
    os.makedirs(os.path.join(results_dir, dataset, "XRAY", "train"))
    os.makedirs(os.path.join(results_dir, dataset, "XRAY", "test"))
    os.makedirs(os.path.join(results_dir, dataset, "CT", "train"))
    os.makedirs(os.path.join(results_dir, dataset, "CT", "test"))
except OSError:
    pass

device = torch.device("cuda:0" if args.cuda else "cpu")

noises_train = torch.randn(len(train_dataset), image_size, image_size) if args.sample_noise and num_xrays == 1 else None
# noises_test = torch.randn(len(valid_loader), image_size, image_size) if args.sample_noise and num_xrays == 1 else None

# create models
if config['discriminator']['mode'] == 'enabled':
    F_net = ResNet_D3D(size=image_size, nc=config['data']['num_ch'], nfilter=32,
                       groupnorm=config['training']['groupnorm'],
                       attention=config['training']['attention'],
                       decode=config['training']['self_sup']).to(device)
else:
    F_net = None
if config['model'] == 'unet':
    from src.unet import UNet3D
    AE = UNet3D(num_xrays, config['data']['num_ch'],
                groupnorm=config['training']['groupnorm'],
                attention=config['training']['attention']).to(device)
elif config['model'] == 'swin-unetr':
    from monai.networks.nets import SwinUNETR
    AE = SwinUNETR(
        img_size=(image_size, image_size, image_size),
        in_channels=num_xrays,
        out_channels=config['data']['num_ch'],
        feature_size=config['transformer']['feature_size'],
        use_checkpoint=config['transformer']['grad_ckpt']).to(device)
else:
    raise Exception(f"Unknown model option {config['model']}")
print(f"Using model {config['model']}")

AE_ema = copy.deepcopy(AE)

# Optimizers
lr = float(config['training']['lr']) * batch_size
b1 = float(config['training']['beta_1'])
b2 = float(config['training']['beta_2'])
wd = float(config['training']['weight_decay'])
f_max_w = float(config['training']['max_d_weight'])
f_weight = float(config['ot']['f_weight'])
emb_spatial_dim = config['autoencoder']['spatial_size']

optim_G = get_optimizer(AE.parameters(), config['training']['optim'], lr=lr, betas=(b1, b2), wd=wd)
if F_net is not None:
    optim_F = get_optimizer(F_net.parameters(),
                            config['training']['optim'],
                            lr=lr,
                            wd=wd)

# load pretrained models if given path to AE
if config['autoencoder']['model_weights_path'] != None:
    AE, AE_ema, optim_G, F_net, optim_F, current_step = load_models_from_checkpoint(config, AE, AE_ema, optim_G, F_net, optim_F)
else:
    # AE.apply(kaiming_weights_init)
    AE_ema = copy.deepcopy(AE).to(device)
    if F_net is not None:
        F_net.apply(kaiming_weights_init)
    current_step = 0

# use activations of a NN as Cost function?
# may be necessary for high-dim spaces
# else will use L2
if config['training']['use_feature_extractor']:
    from src.autoencoder import FeatureExtractor
    C = FeatureExtractor(
        config['data']['num_ch'],
        config['extractor']['base_filters'],
        config['extractor']['n_layers']).to(device)
    C.apply(kaiming_weights_init)
    c_weight = float(config['extractor']['cost_weight'])
    if config['sinkhorn']['mode']:
        from src.optim import ot_activations_difference
        cost = ot_activations_difference
    else:
        from src.optim import activations_difference
        cost = activations_difference
    

g_scaler = None
d_scaler = None

if config['training']['amp']:
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

max_steps = config['training']['steps']
emb_dim = config['autoencoder']['emb_dim']

lr_lambda_g = DecayLR(max_steps, 0, config['training']['decay_steps'],
                      warmup_steps=0, lr=lr).step
lr_lambda_d = DecayLR(max_steps, 0, config['training']['decay_steps'],
                      warmup_steps=0, lr=lr).step
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optim_G,
                                                   lr_lambda=lr_lambda_g)
if F_net is not None:
    lr_scheduler_F = torch.optim.lr_scheduler.LambdaLR(optim_F,
                                                       lr_lambda=lr_lambda_d)

# wandb.watch(AE, log='all', log_freq=10)
# wandb.watch(F_net, log='all', log_freq=1)

pbar = tqdm(range(current_step, max_steps))
for step in pbar:
    # T optimization
    unfreeze(AE)
    if F_net is not None:
        freeze(F_net)
    for it in range(config['training']['t_iters']):
        optim_G.zero_grad()
        data = get_minibatch(train_dataloader, dataloader)
        ct = data["ct"].to(device)
        # if aug_policy is empty it returns the identity
        x = DiffAugment(data["xrays"], config['data']['aug_policy'])
        views = get_views(x, data["view_list"], data["idx"], image_size, noises_train,
                          False, args.rough_align).to(device)
        
        z = torch.randn(ct.size(0), emb_dim, emb_spatial_dim, emb_spatial_dim, emb_spatial_dim).to(device) if config['autoencoder']['add_noise'] else None

        with torch.cuda.amp.autocast(enabled=config['training']['amp']):
            ct_hat = AE(views)

            if config['training']['use_feature_extractor']:
                x, y = C(ct_hat), C(ct)
                recon_loss = c_weight * cost(x, y)          
            else:
                recon_loss = F.mse_loss(ct_hat, ct).mean()
            d_loss = -F_net(ct_hat).mean() if F_net is not None else torch.tensor(0.)
            d_weight = calculate_adaptive_weight(recon_loss, d_loss, AE.D.blocks[-1].weight, f_max_w) if config['training']['adaptive_weight'] else f_max_w
            t_loss = recon_loss + d_weight * d_loss

            if step % 10 == 0 and (it+1) % config['training']['t_iters'] == 0:
                wandb.log({f't_loss' : t_loss.item()}, step=step)
                wandb.log({f'recon_loss' : recon_loss.item()}, step=step)
                wandb.log({f'd_loss' : d_loss.item()}, step=step)
                wandb.log({f'd_weight' : d_weight}, step=step)
                pbar.set_description(
                    f"t_loss: {t_loss.item():.4f}")

        update_model_weights(optim_G, t_loss, config['training']['amp'], g_scaler)

        if step % config['training']['ema_update_every'] == 0 and (it+1) % config['training']['t_iters'] == 0:
            update_ema(AE, AE_ema, float(config['training']['ema_decay']))

    if F_net is not None:
        # f optimization
        freeze(AE)
        unfreeze(F_net)

        data = get_minibatch(train_dataloader, dataloader)
        ct = data["ct"].to(device)
        x = DiffAugment(data["xrays"], config['data']['aug_policy'])
        views = get_views(x, data["view_list"], data["idx"], image_size, noises_train,
                          False, args.rough_align).to(device)

        z = torch.randn(ct.size(0), emb_dim, emb_spatial_dim, emb_spatial_dim, emb_spatial_dim).to(device) if config['autoencoder']['add_noise'] else None

        with torch.cuda.amp.autocast(enabled=config['training']['amp']):
            with torch.no_grad():
                ct_hat = AE(views)

            optim_F.zero_grad()
            if config['training']['self_sup']:
                f_loss, decoding_loss = self_supervised_loss(F_net, ct_hat, ct)
                f_loss = f_weight * f_loss + float(config['training']['self_weight']) * decoding_loss
            else:
                f_loss = f_weight * (F_net(ct_hat).mean() - F_net(ct).mean())

        update_model_weights(optim_F, f_loss, config['training']['amp'], d_scaler)
        wandb.log({f'f_loss' : f_loss.item()}, step=step)
    
    if step % config['training']['plot_freq'] == 0:
        # save ct slices of real and predicted for comparison
        ct_projections = save_ct_slices([ct, ct_hat],
                                        f"{results_dir}/{dataset}/CT/train/real_pred_ct_step_{step}")
        wandb.log({'Train Volume Projections' : [wandb.Image(ct_projections)]}, step=step)
        del ct_projections

    # Test on validation data
    if step % config['testing']['freq'] == 0:
        total_psnr = 0.0
        total_ssim = 0.0
        tpbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for it, test_data in tpbar:
            with torch.no_grad():
                t_ct = test_data["ct"].to(device)
                t_views = get_views(test_data["xrays"], test_data["view_list"], test_data["idx"],
                                    image_size, noises_train,
                                    False, args.rough_align).to(device)

                t_z = torch.randn(t_ct.size(0), emb_dim, emb_spatial_dim, emb_spatial_dim, emb_spatial_dim).to(device) if config['autoencoder']['add_noise'] else None

                t_ct_hat = AE(t_views)

            if it == 0:
                test_ct_projections = save_ct_slices([t_ct, t_ct_hat],
                                                     f"{results_dir}/{dataset}/CT/test/test_real_pred_ct_step_{step}")
                wandb.log({'Test Volume Projections' : [wandb.Image(test_ct_projections)]}, step=step)

            psnr, ssim = eval_psnr_ssim(t_ct[0], t_ct_hat[0])
            total_psnr += psnr
            total_ssim += ssim
        wandb.log({f'Test PSNR' : total_psnr/len(valid_loader)}, step=step)
        wandb.log({f'Test SSIM' : total_ssim/len(valid_loader)}, step=step)
        
        del t_views, t_ct, t_ct_hat; torch.cuda.empty_cache()
      
    # do check pointing & test netG
    if (step+1) % config['training']['save_freq'] == 0:
        torch.save(AE.state_dict(), f"{weights_dir}/AE_step_{step}.pth")
        torch.save(AE_ema.state_dict(), f"{weights_dir}/AE_ema_step_{step}.pth")
        torch.save(optim_G.state_dict(), f"{weights_dir}/AE_optim_step_{step}.pth")
        if F_net is not None:
            torch.save(F_net.state_dict(), f"{weights_dir}/F_step_{step}.pth")
            torch.save(optim_F.state_dict(), f"{weights_dir}/F_optim_step_{step}.pth")

    # Update learning rates
    lr_scheduler_G.step()
    if F_net is not None:
        lr_scheduler_F.step()
