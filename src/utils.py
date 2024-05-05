import os
import random
import torch
import numpy as np
import SimpleITK as sitk
from torch import nn
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from skimage.metrics import structural_similarity as calc_ssim
from torch.utils import data
from scipy import ndimage
from einops import rearrange
from kornia.augmentation import (
    RandomHorizontalFlip,
    RandomRotation,
    RandomRotation3D
)


def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32 if 32 < in_channels else 1, num_channels=in_channels, eps=1e-6, affine=True)


class AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = normalize(in_channels)
        self.qkv = nn.Linear(in_channels, in_channels*3)
        self.proj_out = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        x_in = x
        x = self.norm(x)
        _, c, h, w, d = x.shape
        x = rearrange(x, "b c h w d -> b (h w d) c")
        q, k, v = self.qkv(x).chunk(3, dim=2)

        # compute attention
        attn = (q @ k.transpose(-2, -1)) * (int(c)**(-0.5))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = self.proj_out(out)
        out = rearrange(out, "b (h w d) c -> b c h w d", h=h, w=w, d=d)

        return x_in + out


def eval_psnr_ssim(x, y):
    # tensors -> np arrays -> Unnormalizing
    ct_gt_np, recon_ct_np = to_unNorm(x, y)

    ssim = Structural_Similarity(ct_gt_np, recon_ct_np, size_average=False, PIXEL_MAX=1.0)

    # to Hounsfield scale
    recon_ct_np = toUnMinMax(recon_ct_np).astype(np.int32) - 1024
    ct_gt_np = toUnMinMax(ct_gt_np).astype(np.int32) - 1024

    psnr = Peak_Signal_to_Noise_Rate_3D(ct_gt_np, recon_ct_np, size_average=False, PIXEL_MAX=4095)

    psnr_value = float(psnr)
    ssim_value = float(ssim[-1])

    return psnr_value, ssim_value


def load_models_from_checkpoint(config, AE, AE_ema, AE_optim, F_net, F_optim):
    AE = load_model(AE, config['autoencoder']['model_weights_path'])
    AE_ema = load_model(AE_ema, config['autoencoder']['ema_weights_path'])
    AE_optim = load_model(AE_optim, config['autoencoder']['optim_path'])
    F_net = load_model(F_net, config['discriminator']['model_weights_path'])
    F_optim = load_model(F_optim, config['discriminator']['optim_path'])
    for j in os.path.basename(os.path.splitext(config['autoencoder']['model_weights_path'])[0]).split('_'):
        if j.isdigit():
            current_step = int(j) + 1
    return AE, AE_ema, AE_optim, F_net, F_optim, current_step


def load_cycle_models_from_checkpoint(config, x2c, x2c_ema, c2x, c2x_ema, g_optim, F_x, F_c, F_optim):
    x2c = load_model(x2c, config['autoencoder']['model_weights_path'])
    x2c_ema = load_model(x2c_ema, config['autoencoder']['ema_weights_path'])
    c2x = load_model(c2x, config['autoencoder']['model_weights_path'])
    c2x_ema = load_model(c2x_ema, config['autoencoder']['ema_weights_path'])
    g_optim = load_model(g_optim, config['autoencoder']['optim_path'])
    F_x = load_model(F_x, config['discriminator']['model_weights_path'])
    F_c = load_model(F_c, config['discriminator']['model_weights_path'])
    F_optim = load_model(F_optim, config['discriminator']['optim_path'])
    for j in os.path.basename(os.path.splitext(config['autoencoder']['model_weights_path'])[0]).split('_'):
        if j.isdigit():
            current_step = int(j) + 1
    return x2c, x2c_ema, c2x, c2x_ema, g_optim, F_x, F_c, F_optim, current_step


def load_model(model, path):
    assert os.path.exists(path), f"Path to weight {path} not found!"
    model.load_state_dict(torch.load(path))
    print(f"Loading model from checkpoint: {os.path.basename(path)}")
    return model


def get_minibatch(data_iter, dataloader):
    try:
        minibatch = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        minibatch = next(data_iter)
    return minibatch


def get_views(x, view_list, idx=None, size=64, noises=None, return_tuple=False, rough_align=False):
    if rough_align:
        return get_3d_from_views(x, view_list, x.device, size)
    if len(view_list) == 1 and noises is not None:
        xray = expand_dim(x[:,0,...])
        noises = expand_dim(noises[idx]).repeat(1,1,size-1,1,1).to(x.device)
        if return_tuple:
            return tuple([xray, noises])
        return torch.cat([xray, noises], 2)

    # 2d input views are aligned so they better 'match' GT data
    data = []
    for i, view in enumerate(view_list):
        v = view[0]
        xray = x[:,i,...]
        if v == 2 or 6:
            data.append(torch.transpose((xray.unsqueeze(-1).repeat(1,1,1,size).unsqueeze(1)), 2, 4))
        else:
            data.append(xray.unsqueeze(-1).repeat(1,1,1,size).unsqueeze(1))
    if return_tuple:
        return tuple(data)
    return torch.cat(data, dim=1)


def get_3d_from_views(x, view_list, device, size=64, return_tuple=False):
    data = []
    size = size*len(view_list) if return_tuple else size 
    rotations = [((0, 0), (0, 0), (90, 90)),
                 ((0, 0), (0, 0), (45, 45)),
                 ((0, 0), (0, 0), (0, 0)),
                 ((0, 0), (0, 0), (-45, -45)),
                 ((0, 0), (0, 0), (-90, -90)),
                 ((0, 0), (0, 0), (-135, -135)),
                 ((0, 0), (0, 0), (-180, -180)),
                 ((0, 0), (0, 0), (135, 135))]

    for i, view in enumerate(view_list):
        v = view[0]
        xray = x[:, i, ...]

        if v >= 0 and v < len(rotations):
            rot = RandomRotation3D(degrees=rotations[v], p=1.0, keepdim=True)
            data.append(rot(xray.unsqueeze(-1).repeat(1,1,1,size).unsqueeze(1)))
    if return_tuple:
        return tuple(data)
    return torch.cat(data, dim=1)   


def get_inputs_from_views(x, view_list, img_size):
    x1 = get_3d_from_views(x, view_list[:1], img_size)
    x2 = get_3d_from_views(x, view_list[1:], img_size)

    return x1, x2


def expand_dim(x):
    return x.unsqueeze(1).unsqueeze(1)

def t_repeat(x, size):
    return torch.transpose(x, 2, 4).repeat(1,1,1,1,size)

def u_repeat(x, size):
    return x.unsqueeze(1).repeat(1,1,size,1,1)

def out_projection(x, dim):
    return torch.clamp(torch.unsqueeze(torch.mean(torch.abs(x), dim=dim), 0), 0, 1)

def normal_out_projection(x, dim):
    return toUnnormalize(torch.unsqueeze(torch.mean(x, dim=dim), 0), 0., 1.)

def out_max_projection(x, dim):
    return torch.clamp(torch.unsqueeze(x.max(dim=dim)[0], 0), 0., 1)

def expand_depth(x):
    N, C, H, W = x.size()
    return x.unsqueeze(2).expand(N, C, H, H, W)


def logit_sigmoid(x):
    return -F.softplus(x) + F.softplus(x)

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))

def conv3d(*args, **kwargs):
    return spectral_norm(nn.Conv3d(*args, **kwargs))

def convTranspose3d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose3d(*args, **kwargs))

def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]


def crop_image_by_part3D(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,hw:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,:hw,hw:]
    if part==2:
        return image[:,:,hw:,hw:,:hw]
    if part==3:
        return image[:,:,:hw,hw:,hw:]


def rand_slice(im_size):
    shift = im_size // 7
    return random.randint(shift, im_size - shift)


def Slice(x, n):
    return x[:,n:n+1,:,:]


def reshape_ct(x, n, size=128):
    return Slice(torch.squeeze(F.interpolate(torch.unsqueeze(x, 0),
                                                 size), dim=0), n)


def kaiming_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def xavier_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, gain=1.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def InfiniteSampler(n):
    """Data sampler"""
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    """Data sampler wrapper"""
    def __init__(self, data_source, factor):
        self.num_samples = len(data_source)
        self.factor = factor

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** self.factor


# ct transformations:
# taken from:
# https://github.com/kylekma/X2CT

class Permute(object):
    '''
    Permute
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    def __init__(self, loc):
        self.loc = loc

    def __call__(self, img):
        img = np.transpose(img, self.loc)

        return img


class Resize_image(object):
    '''
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    def __init__(self, size=(3,256,256)):
        if not _isArrayLike(size):
            raise ValueError('each dimension of size must be defined')
        self.size = np.array(size, dtype=np.float32)

    def __call__(self, img):
        z, x, y = img.shape
        ori_shape = np.array((z, x, y), dtype=np.float32)
        resize_factor = self.size / ori_shape
        img_copy = ndimage.interpolation.zoom(img, resize_factor, order=1)

        return img_copy


class Limit_Min_Max_Threshold(object):
    '''
    Restrict in value range. value > max = max,
    value < min = min
    img: 3D, (z, y, x) or (D, H, W)
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    def __init__(self, window_center, window_size):
        self.min = window_center - window_size / 2
        self.max = window_center + window_size / 2

    def __call__(self, img):
        img_copy = img.copy()
        img_copy[img_copy > self.max] = self.max
        img_copy[img_copy < self.min] = self.min
        img_copy = img_copy - self.min

        return img_copy


class Identity(object):
    def __call__(self, img):
        return img


class Normalization(object):
    '''
    To value range -1 - 1
    img: 3D, (z, y, x) or (D, H, W)
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    def __init__(self, min, max, f16=False, round_v=6):
        '''
        :param min:
        :param max:
        :param round_v:
          decrease calculating time
        '''
        if f16:
            self.range = np.array((min, max), dtype=np.float16)
        else:
            self.range = np.array((min, max), dtype=np.float32)
        self.round_v = round_v

    def __call__(self, img):
        img_copy = img.copy()
        img_copy = np.round((img_copy - (self.range[0])) / (self.range[1] - self.range[0]), self.round_v)

        return img_copy


class Normalization_min_max(object):
    '''
    To value range min, max
    img: 3D, (z, y, x) or (D, H, W)
    remove_noise: Set true for baggage data
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    def __init__(self, min_v, max_v, eps=1e-4, remove_noise=False):
        self.max = max_v
        self.min = min_v
        self.eps = eps
        self.remove_noise = remove_noise

    def __call__(self, img):
        # Removing noise from xray machine
        if self.remove_noise:
            img[img < 200] = 0
        img_min = np.min(img)
        img_max = np.max(img)

        img_out = (self.max - self.min) * (img - img_min) / (img_max - img_min + self.eps) + self.min
        return img_out


class Normalization_gaussian(object):
    '''
    To value range 0-1
    img: 3D, (z, y, x) or (D, H, W)
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img_copy = img.copy()
        img_copy = (img_copy - self.mean) / self.std

        return img_copy


class AddGaussianNoise(object):
    def __init__(self, a_min, a_max):
        self.a_min = a_min
        self.a_max = a_max
    def __call__(self, img):
        c, w, h = img.shape
        return np.clip(img + np.random.randn(c, w, h), self.a_min, self.a_max)


class ToTensor(object):
    '''
    To Torch Tensor
    img: 3D, (z, y, x) or (D, H, W)
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    def __init__(self, f16=True):
        self.f16 = f16

    def __call__(self, img):
        if self.f16:
            return torch.from_numpy(img.astype(np.float16))
        return torch.from_numpy(img.astype(np.float32))


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


def toUnnormalize(img, mean, std):
    img = img * std + mean
    return img


def save_ct_slices(list_imgs, save_dir, max_projection=False):
    projections = []
    projection = out_max_projection if max_projection else out_projection
    for img in list_imgs:
        in_data = img[0].clone().cpu()

        in_data_proj_0 = projection(in_data, 1)
        in_data_proj_1 = projection(in_data, 3)
        in_data_proj_2 = projection(in_data, 2)

        projections.append(in_data_proj_0)
        projections.append(in_data_proj_1)
        projections.append(in_data_proj_2)
    
    save_ct = vutils.make_grid(torch.cat(projections,0),
                               nrow=3, normalize=False, scale_each=False)
    vutils.save_image(save_ct, f"{save_dir}"+'.png')
    return save_ct


def save_volume(nparray, name, mha=True):
    np.save(name + '.npy', nparray)
    nparray.squeeze(0).astype('int16').tofile(name + '.raw')
    if mha:
        save_mha(nparray.squeeze(0), spacing=(1., 1., 1.), origin=(0, 0, 0), path=name+'.mha')


# For visualising using e.g 3DSlicer
def save_mha(volume, spacing, origin, path):
    itkimage = sitk.GetImageFromArray(volume, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, path, True)


def save_xrays(real_xrays, pred_xrays, save_dir):
    pred = torch.unsqueeze(pred_xrays[0].clone().cpu(), 1)
    pred = clamp_tensor(toUnnormalize(pred, 0., 1.))
    
    real = torch.unsqueeze(real_xrays[0].clone().cpu(), 1)
    real = clamp_tensor(toUnnormalize(real, 0., 1.))
    
    save = vutils.make_grid(torch.cat([real, pred], 0),
                            normalize=False, scale_each=False)
    vutils.save_image(save, f"{save_dir}"+".png",
                      nrow=2)
    return save


def save_D_decodings(real_big, pred_big, real_part, pred_part, name):
    save_big_real = clamp_tensor(toUnnormalize(real_big[0].clone().cpu().unsqueeze(1), 0., 1.))
    save_big_pred = clamp_tensor(toUnnormalize(pred_big[0].clone().cpu().unsqueeze(1), 0., 1.))
    save_part_real = clamp_tensor(toUnnormalize(real_part[0].clone().cpu().unsqueeze(1), 0., 1.))
    save_part_pred = clamp_tensor(toUnnormalize(pred_part[0].clone().cpu().unsqueeze(1), 0., 1.))
    save_decoded = vutils.make_grid(torch.cat([save_big_real, save_big_pred,
                                               save_part_real, save_part_pred],
                                              0), normalize=False)
    vutils.save_image(save_decoded,
                      f"{name}"+".png",
                      nrow=3)
    return save_decoded

def clamp_tensor(t, vmin=0, vmax=1):
    return torch.clamp(t, vmin, vmax)
    
def MAE(arr1, arr2, size_average=True):
    '''
    :param arr1:
      Format-[NDHW], OriImage
    :param arr2:
      Format-[NDHW], ComparedImage
    :return:
      Format-None if size_average else [N]
    '''
    assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
    assert (arr1.ndim == 4) and (arr2.ndim == 4)
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    if size_average:
        return np.abs(arr1 - arr2).mean()
    else:
        return np.abs(arr1 - arr2).mean(1).mean(1).mean(1)


def MSE(arr1, arr2, size_average=True):
    '''
    :param arr1:
    Format-[NDHW], OriImage
    :param arr2:
    Format-[NDHW], ComparedImage
    :return:
    Format-None if size_average else [N]
    '''
    assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
    assert (arr1.ndim == 4) and (arr2.ndim == 4)
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    if size_average:
        return np.power(arr1 - arr2, 2).mean()
    else:
        return np.power(arr1 - arr2, 2).mean(1).mean(1).mean(1)


def Peak_Signal_to_Noise_Rate_3D(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
  '''
  :param arr1:
    Format-[NDHW], OriImage [0,1]
  :param arr2:
    Format-[NDHW], ComparedImage [0,1]
  :return:
    Format-None if size_average else [N]
  '''
  assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
  assert (arr1.ndim == 4) and (arr2.ndim == 4)
  arr1 = arr1.astype(np.float64)
  arr2 = arr2.astype(np.float64)
  eps = 1e-10
  se = np.power(arr1 - arr2, 2)
  mse = se.mean(axis=1).mean(axis=1).mean(axis=1)
  zero_mse = np.where(mse == 0)
  mse[zero_mse] = eps
  psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
  # #zero mse, return 100
  psnr[zero_mse] = 100

  if size_average:
    return psnr.mean()
  else:
    return psnr


def Peak_Signal_to_Noise_Rate(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
    '''
    :param arr1:
    Format-[NDHW], OriImage [0,1]
    :param arr2:
    Format-[NDHW], ComparedImage [0,1]
    :return:
    Format-None if size_average else [N]
    '''
    assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
    assert (arr1.ndim == 4) and (arr2.ndim == 4)
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    eps = 1e-10
    se = np.power(arr1 - arr2, 2)
    # Depth
    mse_d = se.mean(axis=2, keepdims=True).mean(axis=3, keepdims=True).squeeze(3).squeeze(2)
    zero_mse = np.where(mse_d==0)
    mse_d[zero_mse] = eps
    psnr_d = 20 * np.log10(PIXEL_MAX / np.sqrt(mse_d))
    # #zero mse, return 100
    psnr_d[zero_mse] = 100
    psnr_d = psnr_d.mean(1)

    # Height
    mse_h = se.mean(axis=1, keepdims=True).mean(axis=3, keepdims=True).squeeze(3).squeeze(1)
    zero_mse = np.where(mse_h == 0)
    mse_h[zero_mse] = eps
    psnr_h = 20 * np.log10(PIXEL_MAX / np.sqrt(mse_h))
    # #zero mse, return 100
    psnr_h[zero_mse] = 100
    psnr_h = psnr_h.mean(1)

    # Width
    mse_w = se.mean(axis=1, keepdims=True).mean(axis=2, keepdims=True).squeeze(2).squeeze(1)
    zero_mse = np.where(mse_w == 0)
    mse_w[zero_mse] = eps
    psnr_w = 20 * np.log10(PIXEL_MAX / np.sqrt(mse_w))
    # #zero mse, return 100
    psnr_w[zero_mse] = 100
    psnr_w = psnr_w.mean(1)

    psnr_avg = (psnr_h + psnr_d + psnr_w) / 3
    if size_average:
        return [psnr_d.mean(), psnr_h.mean(), psnr_w.mean(), psnr_avg.mean()]
    return [psnr_d, psnr_h, psnr_w, psnr_avg]


def Structural_Similarity(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
    '''
    :param arr1:
    Format-[NDHW], OriImage [0,1]
    :param arr2:
    Format-[NDHW], ComparedImage [0,1]
    :return:
    Format-None if size_average else [N]
    '''
    assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
    assert (arr1.ndim == 4) and (arr2.ndim == 4)
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)

    N = arr1.shape[0]
    # Depth
    arr1_d = np.transpose(arr1, (0, 2, 3, 1))
    arr2_d = np.transpose(arr2, (0, 2, 3, 1))
    ssim_d = []
    for i in range(N):
        ssim = calc_ssim(arr1_d[i], arr2_d[i], data_range=PIXEL_MAX, channel_axis=1)
        ssim_d.append(ssim)
    ssim_d = np.asarray(ssim_d, dtype=np.float64)

    # Height
    arr1_h = np.transpose(arr1, (0, 1, 3, 2))
    arr2_h = np.transpose(arr2, (0, 1, 3, 2))
    ssim_h = []
    for i in range(N):
        ssim = calc_ssim(arr1_h[i], arr2_h[i], data_range=PIXEL_MAX, channel_axis=1)
        ssim_h.append(ssim)
    ssim_h = np.asarray(ssim_h, dtype=np.float64)

    # Width
    # arr1_w = np.transpose(arr1, (0, 1, 2, 3))
    # arr2_w = np.transpose(arr2, (0, 1, 2, 3))
    ssim_w = []
    for i in range(N):
        ssim = calc_ssim(arr1[i], arr2[i], data_range=PIXEL_MAX, channel_axis=1)
        ssim_w.append(ssim)
    ssim_w = np.asarray(ssim_w, dtype=np.float64)

    ssim_avg = (ssim_d + ssim_h + ssim_w) / 3

    if size_average:
        return [ssim_d.mean(), ssim_h.mean(), ssim_w.mean(), ssim_avg.mean()]
    else:
        return [ssim_d, ssim_h, ssim_w, ssim_avg]


def to_unNorm(ct, pred_ct, inverse=False):
    fake_ct = pred_ct.clone().detach().cpu().numpy()
    real_ct = ct.clone().detach().cpu().numpy()

    fake_ct_t = np.transpose(fake_ct, (0, 2, 1, 3))
    real_ct_t = np.transpose(real_ct, (0, 2, 1, 3))

    if inverse:
        fake_ct_t = fake_ct_t[:, ::-1, :, :]
        real_ct_t = real_ct_t[:, ::-1, :, :]

    fake_ct_t = toUnnormalize(fake_ct_t, 0., 1.)
    real_ct_t = toUnnormalize(real_ct_t, 0., 1.)

    fake_ct_t = np.clip(fake_ct_t, 0, 1)

    return real_ct_t, fake_ct_t


def save_volume(nparray, name, mha=True):
    nparray.squeeze(0).astype('int16').tofile(name + '.raw')
    if mha:
        save_mha(nparray.squeeze(0), spacing=(1., 1., 1.), origin=(0, 0, 0), path=name+'.mha')


def save_mha(volume, spacing, origin, path):
    itkimage = sitk.GetImageFromArray(volume, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, path, True)


def toUnMinMax(input_image, min=0, max=2000):
    image = input_image * (max - min) + min
    return image


def update_config(config, unknown):
    # update config given args
    for idx,arg in enumerate(unknown):
        if arg.startswith("--"):
            if (':') in arg:
                k1,k2 = arg.replace("--","").split(':')
                argtype = type(config[k1][k2])
                if argtype == bool:
                    v = unknown[idx+1].lower() == 'true'
                else:
                    if config[k1][k2] is not None:
                        v = type(config[k1][k2])(unknown[idx+1])
                    else:
                        v = unknown[idx+1]
                print(f'Changing {k1}:{k2} ---- {config[k1][k2]} to {v}')
                config[k1][k2] = v
            else:
                k = arg.replace('--','')
                v = unknown[idx+1]
                argtype = type(config[k])
                print(f'Changing {k} ---- {config[k]} to {v}')
                config[k] = v

    return config


def save_config(outpath, config):
    from yaml import safe_dump
    with open(outpath, 'w') as f:
        safe_dump(config, f)


def calculate_adaptive_weight(recon_loss, g_loss, last_layer, disc_weight_max):
    recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()
    return d_weight
