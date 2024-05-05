import numpy as np
from torch import nn
import random
from .utils import normalize, AttnBlock3D


class ResNet_D3D(nn.Module):
    "Discriminator ResNet architecture from https://github.com/harryliew/WGAN-QC"
    "Adapted for 3D"
    def __init__(self, size=64, nc=1, nfilter=64, nfilter_max=512, res_ratio=0.1,
                 groupnorm=False, attention=False, decode=False):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max
        self.nc = nc
        self.attn = attention
        self.decode = decode

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        nf0 = min(nf, nf_max)
        nf1 = min(nf * 2, nf_max)
        blocks = [
            ResNetBlock3D(nf0, nf0, bn=False, res_ratio=res_ratio, groupnorm=groupnorm),
            ResNetBlock3D(nf0, nf1, bn=False, res_ratio=res_ratio, groupnorm=groupnorm)
        ]

        for i in range(1, nlayers+1):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool3d(3, stride=2, padding=1),
                ResNetBlock3D(nf0, nf0, bn=False, res_ratio=res_ratio, groupnorm=groupnorm),
                ResNetBlock3D(nf0, nf1, bn=False, res_ratio=res_ratio, groupnorm=groupnorm),
            ]
            if i == nlayers and self.attn:
                blocks.append(AttnBlock3D(nf1))

        self.conv_img = nn.Conv3d(nc, 1*nf, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0*s0*s0*s0, 1)

        if self.decode:
            from .decoder import SimpleDecoder
            self.dec_big = SimpleDecoder(512, nc)
            self.dec_part = SimpleDecoder(32, nc)

    def forward(self, x, is_pred=True):
        batch_size = x.size(0)

        out_1 = self.relu((self.conv_img(x)))
        out_2 = self.resnet(out_1)

        if self.decode and not is_pred:
            rec_img = self.dec_big(out_2)

            part = random.randint(0, 3)
            rec_img_part = None
            if part==0:
                rec_img_part = self.dec_part(out_1[...,:8,:8,:8])
            if part==1:
                rec_img_part = self.dec_part(out_1[...,:8,8:,:8])
            if part==2:
                rec_img_part = self.dec_part(out_1[...,8:,:8,:8])
            if part==3:
                rec_img_part = self.dec_part(out_1[...,8:,8:,:8])

            out_2 = out_2.view(batch_size, self.nf0*self.s0*self.s0*self.s0) #[B,32768]
            out_2 = self.fc(out_2)

            return out_2, rec_img, rec_img_part, part

        out_2 = out_2.view(batch_size, self.nf0*self.s0*self.s0*self.s0) #[B,32768]
        out_2 = self.fc(out_2)

        return out_2


class ResNetBlock3D(nn.Module):
    def __init__(self, fin, fout, fhidden=None, bn=True, res_ratio=0.1, groupnorm=False):
        super().__init__()
        # Attributes
        self.bn = bn
        self.is_bias = not bn
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden
        self.res_ratio = res_ratio

        # Submodules
        self.conv_0 = nn.Conv3d(self.fin, self.fhidden, 3, stride=1, padding=1, bias=self.is_bias)
        if self.bn:
            self.bn2d_0 = normalize(self.fhidden) if groupnorm  else nn.BatchNorm3d(self.fhidden)
        self.conv_1 = nn.Conv3d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=self.is_bias)
        if self.bn:
            self.bn2d_1 = normalize(self.fout) if groupnorm  else nn.BatchNorm3d(self.fout)
        if self.learned_shortcut:
            self.conv_s = nn.Conv3d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)
            if self.bn:
                self.bn2d_s = normalize(self.fout) if groupnorm else nn.BatchNorm3d(self.fout)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(x)
        if self.bn:
            dx = self.bn2d_0(dx)
        dx = self.relu(dx)
        dx = self.conv_1(dx)
        if self.bn:
            dx = self.bn2d_1(dx)
        out = self.relu(x_s + self.res_ratio*dx)
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
            if self.bn:
                x_s = self.bn2d_s(x_s)
        else:
            x_s = x
        return x_s
