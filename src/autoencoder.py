import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

## These classes taken and adapted from 'Unaligned 2D to 3D Translation with Conditional
## Vector-Quantized Code Diffusion using Transformers'


class AutoEncoder2D3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_ch = config['autoencoder']['emb_dim']*2 if config['autoencoder']['add_noise'] else config['autoencoder']['emb_dim']
        self.E1 = Encoder2D(
            in_channels=config['data']['num_ch'],
            nf=config['autoencoder']['enc_nf'],
            out_channels=config['autoencoder']['emb_dim'],
            ch_mult=config['autoencoder']['enc_ch_mult'],
            num_res_blocks=config['autoencoder']['num_res_blocks'],
            resolution=config['data']['xray_scale'],
            attn_resolutions=config['autoencoder']['enc_attn_resolutions'])

        self.E2 = Encoder2D(
            in_channels=config['data']['num_ch'],
            nf=config['autoencoder']['enc_nf'],
            out_channels=config['autoencoder']['emb_dim'],
            ch_mult=config['autoencoder']['enc_ch_mult'],
            num_res_blocks=config['autoencoder']['num_res_blocks'],
            resolution=config['data']['xray_scale'],
            attn_resolutions=config['autoencoder']['enc_attn_resolutions'])

        self.D = Decoder3D(
            in_channels=in_ch,
            out_channels=config['data']['num_ch'],
            nf=config['autoencoder']['dec_nf'],
            ch_mult=config['autoencoder']['dec_ch_mult'],
            num_res_blocks=config['autoencoder']['num_res_blocks'],
            resolution=config['data']['image_size'],
            attn_resolutions=config['autoencoder']['dec_attn_resolutions'],
            resblock_name=config['autoencoder']['resblock_name'])

        if config['autoencoder']['add_noise']:
            self.emb_dim = config['autoencoder']['emb_dim']
        self.add_noise = config['autoencoder']['add_noise']

        self.expand_features = config['training']['features_mode']

    def train_step(self, x1, x2, z):
        z1 = self.E1(x1)
        z2 = self.E2(x2)
            
        if self.expand_features == 'expand':
            b, c, h, w = z1.size()
            zm = (z1 + z2).unsqueeze(4).expand(b, c, h, h, w)
        else:
            zm = torch.reshape((z1 + z2), (b, -1, h//2, h//2, h//2))

        if self.add_noise:
            zm = torch.cat([zm, z], dim=1)

        ct_hat = self.D(zm)

        return ct_hat

    def encode(self, x1, x2):
        z1 = self.E1(x1)
        z2 = self.E2(x2)
            
        if self.expand_features == 'expand':
            b, c, h, w = z1.size()
            zm = (z1 + z2).unsqueeze(4).expand(b, c, h, h, w)
        else:
            zm = torch.reshape((z1 + z2), (b, -1, h//2, h//2, h//2))

        return zm

    def decode(self, zm):
        return self.D(zm)


class Encoder2D(nn.Module):
    def __init__(self, in_channels, nf, out_channels, ch_mult, num_res_blocks, resolution, attn_resolutions):
        super().__init__()
        num_resolutions = len(ch_mult)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)

        blocks = []
        # initial convoltion
        blocks.append(nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1))

        # residual and downsampling blocks, with attention on smaller res
        for i in range(num_resolutions):
            block_in_ch = nf * in_ch_mult[i]
            block_out_ch = nf * ch_mult[i]
            for _ in range(num_res_blocks):
                blocks.append(ResBlock2D(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock2D(block_in_ch))

            if i != num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = curr_res // 2

        # non-local attention block
        blocks.append(ResBlock2D(block_in_ch, block_in_ch))
        blocks.append(AttnBlock2D(block_in_ch))
        blocks.append(ResBlock2D(block_in_ch, block_in_ch))

        # normalise and convert to latent size
        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, out_channels, kernel_size=3, stride=1, padding=1))
            
        self.blocks = nn.ModuleList(blocks)
            

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Decoder3D(nn.Module):
    def __init__(self, in_channels, out_channels, nf, ch_mult, num_res_blocks, resolution, attn_resolutions, resblock_name):
        super().__init__()
        num_resolutions = len(ch_mult)
        block_in_ch = nf * ch_mult[-1]
        curr_res = resolution // 2 ** (num_resolutions-1)

        blocks = []
        # initial conv
        blocks.append(nn.Conv3d(in_channels, block_in_ch, kernel_size=3, stride=1, padding=1))

        resblocks = {
            "resblock": ResBlock3D,
            "depthwise_block": DepthwiseResBlock
        }
        Block = resblocks[resblock_name]

        # non-local attention block
        blocks.append(Block(block_in_ch, block_in_ch))
        blocks.append(AttnBlock3D(block_in_ch))
        blocks.append(Block(block_in_ch, block_in_ch))

        for i in reversed(range(num_resolutions)):
            block_out_ch = nf * ch_mult[i]

            for _ in range(num_res_blocks):
                blocks.append(Block(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock3D(block_in_ch))

            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = curr_res * 2

        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv3d(block_in_ch, out_channels, kernel_size=3, stride=1, padding=1))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class FeatureExtractor(nn.Module):
    # This model is used for computing the cost instead of mse_loss
    def __init__(self, nc, ndf, n_layers=3, activations_only=True):
        super().__init__()

        self.act_only = activations_only
        layers = [nn.Conv3d(nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        ndf_mult = 1
        ndf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            ndf_mult_prev = ndf_mult
            ndf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv3d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(ndf * ndf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        ndf_mult_prev = ndf_mult
        ndf_mult = min(2 ** n_layers, 8)

        layers += [
            nn.Conv3d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(ndf * ndf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # layers += [
        #     nn.Conv3d(ndf * ndf_mult, 1, kernel_size=4, stride=1, padding=1)]  # output 1 channel prediction map
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        activations = []
        for layer in self.main:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                activations.append(x)
        if self.act_only:
            return activations
        x = x.view(-1, self.dim5)
        x = x / torch.norm(x, dim=1, keepdim=True)  # L2 normalization
        return x


class DepthwiseResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = LayerNorm(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.norm2 = LayerNorm(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
        self.conv_out = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.gelu(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)
        
        return x + x_in


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)

        return x


class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in


class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in


class AttnBlock2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.k = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.v = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


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


def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32 if 32 < in_channels else 1, num_channels=in_channels, eps=1e-6, affine=True)


@torch.jit.script
def swish(x):
    return x*torch.sigmoid(x)


class SimpleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), # 64
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.Conv2d(512, 2048, 4, 2, 1),
            nn.BatchNorm2d(2048),
            nn.ELU(),
            nn.Conv2d(2048, 4096, 4, 2, 1),
        )
    def forward(self, x):
        return self.main(x)


class SimpleDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(2048*2, 2048, 4, 2, 1),
            nn.BatchNorm3d(2048),
            nn.ELU(),
            nn.ConvTranspose3d(2048, 512, 4, 2, 1),
            nn.BatchNorm3d(512),
            nn.ELU(),
            nn.ConvTranspose3d(512, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.ELU(),
            nn.ConvTranspose3d(128, 32, 4, 2, 1),
            nn.BatchNorm3d(32),
            nn.ELU(),
            nn.ConvTranspose3d(32, 1, 4, 2, 1),
        )
    def forward(self, x):
        return self.main(x)
