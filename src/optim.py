import torch
import numpy as np
from torch import nn
from math import prod
import torch.nn.functional as F
from geomloss import SamplesLoss


def contrastive_loss(out1, out2, batch_size, temperature=0.5):
    # [2*B, D]
    out1 = F.normalize(out1, dim=-1)
    out2 = F.normalize(out2, dim=-1)
    out = torch.cat([out1, out2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out1 * out2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    return loss
    


def self_supervised_loss(F_net, pred, gt):
    out_pred = F_net(pred)
    out_real, rec_img, rec_part, part = F_net(gt, False)

    f_loss = out_pred.mean() - out_real.mean()
    decode_loss = F.mse_loss(rec_img, F.interpolate(gt, rec_img.shape[-1], mode='trilinear')).mean() +\
        F.mse_loss(rec_part, F.interpolate(crop_image_by_part(gt, part), rec_img.shape[-1], mode='trilinear')).mean()

    return f_loss, decode_loss


def crop_image_by_part(image, part):
    hw = image.shape[-1]//2
    if part==0:
        return image[:,:,:hw,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:,:hw]
    if part==2:
        return image[:,:,hw:,:hw,:hw]
    if part==3:
        return image[:,:,hw:,hw:,:hw]


def update_ema(model, ema_model, ema_rate):
    for p1, p2 in zip(model.parameters(), ema_model.parameters()):
        # Beta * previous ema weights + (1 - Beta) * current non ema weight
        p2.data.mul_(ema_rate)
        p2.data.add_(p1.data * (1 - ema_rate))


def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)

def activations_difference(acts1, acts2):
    val = 0
    for act1, act2 in zip(acts1, acts2):
        act1 = normalize_tensor(act1)
        act2 = normalize_tensor(act2)
        diff = (act1 - act2) ** 2
        diff = diff.mean(dim=(1,2,3,4))
        val += diff
    return torch.mean(val)


def ot_activations_difference(acts1, acts2):
    val = 0
    for act1, act2 in zip(acts1, acts2):
        act1 = normalize_tensor(act1)
        act2 = normalize_tensor(act2)
        diff = ot_loss(act1, act2)
        val += diff
    return val


def ot_loss(x,y):
    ot_loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    return ot_loss_fn(x.view(x.size(0), -1), y.view(y.size(0), -1))


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    
    
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)


class AdaptivePseudoAugment:
    def __init__(self, start_epoch=14, initial_prob=0., threshold=0.2, iterations=4):
        self.start = start_epoch
        self.init_prob = initial_prob
        self.t = threshold # lower value is suggested for smaller datasets
        self.speed = 1e-6
        self.it = iterations
        self.grad_accumulation = 5
        self.max_prob = 0.7
        self.lambda_rs = []
        self.lambda_fs = []

    def update_lambdas(self, batch_size, num_mix_fakes, logits_real, logits_fake):
        _lambda_r = logit_sigmoid(logits_real[:batch_size - num_mix_fakes]).sign().mean().item()
        self.lambda_rs.append(_lambda_r)
        _lambda_f = logit_sigmoid(logits_fake[:batch_size]).sign().mean().item()
        self.lambda_fs.append(_lambda_f)
        
    def adjust_prob(self, batch_size):
        if len(self.lambda_rs) != 0 or len(self.lambda_fs) != 0:
            lambda_r = sum(self.lambda_rs) / len(self.lambda_rs)
            lambda_f = sum(self.lambda_fs) / len(self.lambda_fs)
            lambda_rf = (lambda_r - lambda_f) / 2 # this can be used instead of λr for adjusting
            self.init_prob += \
                np.sign(lambda_rf - self.t) * \
                self.speed * \
                batch_size * self.grad_accumulation * self.it
            self.init_prob = np.clip(self.init_prob, 0., self.max_prob)
            self.lambda_rs = []


def logit_sigmoid(x):
    return -F.softplus(x) + F.softplus(x)

def loss_hinge_d(pred, real):
    if real:
        return F.mish(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean()
    return F.mish(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()


def calculate_adaptive_weight(recon_loss, g_loss, last_layer, disc_weight_max):
    recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()
    return d_weight


def loss_hinge_g(dis_fake):
    return -torch.mean(dis_fake)


def loss_ls_d(pred, real):
    if real:
        return torch.mean((pred + 1).pow(2))
    return torch.mean((pred - 1).pow(2))


def loss_ls_g(dis_fake):
    return torch.mean(dis_fake.pow(2))

# LeCam Regularziation loss
def lecam_reg(dis_real, dis_fake, ema_losses, mode):
    if mode == 'XRAYS':
        ema_fake = ema_losses.D_X_fake
        ema_real = ema_losses.D_X_real
    elif mode == 'CT':
        ema_fake = ema_losses.D_CT_fake
        ema_real = ema_losses.D_CT_real

    reg = torch.mean(F.relu(dis_real - ema_fake).pow(2)) + \
        torch.mean(F.relu(ema_real - dis_fake).pow(2))
    return reg


# Simple wrapper that applies EMA to a model. COuld be better done in 1.0 using
# the parameters() and buffers() module functions, but for now this works
# with state_dicts using .copy_
class ema(object):
    def __init__(self, source, target, decay=0.9999, start_itr=0):
        self.source = source
        self.target = target
        self.decay = decay
        # Optional parameter indicating what iteration to start the decay at
        self.start_itr = start_itr
        # Initialize target's params to be source's
        self.source_dict = self.source.state_dict()
        self.target_dict = self.target.state_dict()
        print('Initializing EMA parameters to be source parameters...')
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.source_dict[key].data)
                # target_dict[key].data = source_dict[key].data # Doesn't work!

    def update(self, itr):
        # If an iteration counter is provided and itr is less than the start itr,
        # peg the ema weights to the underlying weights.
        if itr is None:
            decay = self.decay
        elif itr < self.start_itr:#if itr and itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.target_dict[key].data * decay
                                                 + self.source_dict[key].data * (1 - decay))


# Simple wrapper that applies EMA to losses.
class ema_losses(object):
    def __init__(self, init=1000., decay=0.999, start_itr=0):
        self.G_loss = init
        self.D_X_loss_real = init
        self.D_X_loss_fake = init
        self.D_X_real = init
        self.D_X_fake = init
        self.D_CT_loss_real = init
        self.D_CT_loss_fake = init
        self.D_CT_real = init
        self.D_CT_fake = init
        self.decay = decay
        self.start_itr = start_itr

    def update(self, cur, mode, itr):
        if itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        if mode == 'G_loss':
            self.G_loss = self.G_loss*decay + cur*(1 - decay)
        elif mode == 'D_X_loss_real':
            self.D_X_loss_real = self.D_X_loss_real*decay + cur*(1 - decay)
        elif mode == 'D_X_loss_fake':
            self.D_X_loss_fake = self.D_X_loss_fake*decay + cur*(1 - decay)
        elif mode == 'D_X_real':
            self.D_X_real = self.D_X_real*decay + cur*(1 - decay)
        elif mode == 'D_X_fake':
            self.D_X_fake = self.D_X_fake*decay + cur*(1 - decay)
        elif mode == 'D_CT_loss_real':
            self.D_CT_loss_real = self.D_CT_loss_real*decay + cur*(1 - decay)
        elif mode == 'D_CT_loss_fake':
            self.D_CT_loss_fake = self.D_CT_loss_fake*decay + cur*(1 - decay)
        elif mode == 'D_CT_real':
            self.D_CT_real = self.D_CT_real*decay + cur*(1 - decay)
        elif mode == 'D_CT_fake':
            self.D_CT_fake = self.D_CT_fake*decay + cur*(1 - decay)


def get_optimizer(params, optim, lr, betas=(0.9, 0.999), wd=1e-10):
    if optim == 'adam':
        return torch.optim.Adam(params, lr=lr, betas=betas,
                                weight_decay=wd, eps=1e-3)
    elif optim == 'adamw':
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd, eps=1e-3)
    elif optim == 'adabelief':
        from adabelief_pytorch import AdaBelief
        return AdaBelief(params, lr=lr, betas=betas, print_change_log=False, eps=1e-14,
                         weight_decay=0, weight_decouple=True, rectify=True, fixed_decay=False,
                         amsgrad=False)
    elif optim == 'sgd':
        return torch.optim.SGD(params, lr=lr)
    else:
        raise Exception(f"Optimizer option {optim} not supported!")


class DecayLR:
    def __init__(self, epochs, offset, decay_epochs, warmup_steps, lr):
        epoch_flag = epochs - decay_epochs
        assert (epoch_flag > 0), "Decay must start before the training session ends!"
        self.epochs = epochs
        self.offset = offset
        self.decay_epochs = decay_epochs
        self.lr = lr
        self.warmup_steps = warmup_steps

    def step(self, epoch):
        if epoch <= self.warmup_steps:
            return float(epoch+1) * self.warmup_steps
        else:
            return 1.0 - max(0, epoch + self.offset - self.decay_epochs) / (
                self.epochs - self.decay_epochs)


class OptimWeight:
    def __init__(self, start, stop, max_weight, inc=0.0):
        self.start = start
        self.stop = stop
        self.max_weight = max_weight

        if inc == 0.0:
            if self.stop == 1:
                self.inc = self.max_weight
            else:
                self.inc = self.max_weight / self.stop
        else:
            self.inc = inc

    def inc_step(self, epoch, weight):
        if epoch >= self.start and weight < self.max_weight:
            weight += self.inc
        return weight

    def dec_step(self, epoch, weight):
        if epoch >= self.start and weight > self.max_weight:
            weight -= self.inc
        return weight

    
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        y_size = y.size()
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss/prod(y_size)


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)
