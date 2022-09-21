from layer import *

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
import random 
class UNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm'):
        super(UNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.enc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc5 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc6 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc7 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc8 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=[])
        #self.transformer = SpatialAttention(kernel_size=3)

        self.dec8 = DECNR2d(1 * 8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=0.5)
        self.dec7 = DECNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=0.5)
        self.dec6 = DECNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=0.5)
        self.dec5 = DECNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=[])
        self.dec4 = DECNR2d(2 * 8 * self.nch_ker, 4 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=[])
        self.dec3 = DECNR2d(2 * 4 * self.nch_ker, 2 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=[])
        self.dec2 = DECNR2d(2 * 2 * self.nch_ker, 1 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=[])
        self.dec1 = DECNR2d(2 * 1 * self.nch_ker, 1 * self.nch_out, stride=2, norm=[],        relu=[],  drop=[], bias=False)

    def forward(self, x):

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        enc7 = self.enc7(enc6)
        enc8 = self.enc8(enc7)

        dec8 = self.dec8(enc8)
        dec7 = self.dec7(torch.cat([enc7, dec8], dim=1))
        dec6 = self.dec6(torch.cat([enc6, dec7], dim=1))
        dec5 = self.dec5(torch.cat([enc5, dec6], dim=1))
        dec4 = self.dec4(torch.cat([enc4, dec5], dim=1))
        dec3 = self.dec3(torch.cat([enc3, dec4], dim=1))
        dec2 = self.dec2(torch.cat([enc2, dec3], dim=1))
        dec1 = self.dec1(torch.cat([enc1, dec2], dim=1))

        x = torch.tanh(dec1)

        return x




class ResNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm', nblk=4):
        super(ResNet, self).__init__()


        self.channels = nch_in
        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm
        self.nblk = nblk


        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True
        self.enc1_c = CNR2d(self.nch_in,      32, kernel_size=7, stride=1, padding=3, norm=self.norm, relu=0.0)
        self.enc2_c = CNR2d(32, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)
        self.enc3_c = CNR2d(self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)
        self.enc4_c = CNR2d(2 *self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.enc1_r = CNR2d(self.nch_in,      32, kernel_size=7, stride=1, padding=3, norm=self.norm, relu=0.0)
        self.enc2_r = CNR2d(32, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)
        self.enc3_r = CNR2d(self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)
        self.enc4_r = CNR2d(2 *self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)


        self.cbam_cat = CBAM(n_channels_in = self.nch_ker*8,reduction_ratio = 16, kernel_size = 3)
        self.cbam_c = CBAM(n_channels_in = self.nch_ker*4,reduction_ratio = 16, kernel_size = 3)
        self.cbam_r = CBAM(n_channels_in = self.nch_ker*4,reduction_ratio = 16, kernel_size = 3)

        if self.nblk:
            res = []

            for i in range(self.nblk):
                res += [ResBlock(8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0, padding_mode='reflection')]

            self.res = nn.Sequential(*res)

        self.dec5 = DECNR2d(16 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)
        
        self.dec4 = DECNR2d(8 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.dec3 = DECNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.dec2 = DECNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=7, stride=1, padding=3, norm=self.norm, relu=0.0)        

        self.dec1 = CNR2d(1 * self.nch_ker, self.nch_out, kernel_size=7, stride=1, padding=3, norm=[], relu=[], bias=False)

    def forward(self, x,y):
        x = self.enc1_c(x)
        x = self.enc2_c(x)
        x = self.enc3_c(x)
        x = self.enc4_c(x)

        x,_ = self.cbam_c(x)

        y = self.enc1_r(y)
        y = self.enc2_r(y)
        y = self.enc3_r(y)
        y = self.enc4_r(y)

        y,_ = self.cbam_r(y)

        xy =torch.cat((x,y),1)

        cbam_xy,sparse_map = self.cbam_cat(xy)

        if self.nblk:
            r_xy = self.res(xy)
        rcbam_xy = torch.cat((r_xy,cbam_xy),1) 

        rcbam_xy = self.dec5(rcbam_xy)
        rcbam_xy = self.dec4(rcbam_xy)
        rcbam_xy = self.dec3(rcbam_xy)
        rcbam_xy = self.dec2(rcbam_xy)
        rcbam_xy = self.dec1(rcbam_xy)

        rcbam_xy = torch.tanh(rcbam_xy)

        return rcbam_xy,sparse_map


# class SCFT(nn.Module):
#     def __init__(self, d_channel=448):
#         super(SCFT, self).__init__()
        
#         self.W_v = nn.Parameter(torch.randn(d_channel, d_channel)) # [448, 448]
#         self.W_k = nn.Parameter(torch.randn(d_channel, d_channel)) # [448, 448]
#         self.W_q = nn.Parameter(torch.randn(d_channel, d_channel)) # [448, 448]
#         self.coef = d_channel ** .5
        
#         self.gamma = 12.
    
#     def forward(self, V_r, V_s):
        
#         wq_vs = torch.matmul(self.W_q, V_s) # [1, 448, 1024]
#         wk_vr = torch.matmul(self.W_k, V_r).permute(0, 2, 1) # [1, 448, 1024]
#         alpha = F.softmax(torch.matmul(wq_vs, wk_vr) / self.coef, dim=-1) # Eq.(2)
        
#         wv_vr = torch.matmul(self.W_v, V_r)
#         v_asta = torch.matmul(alpha, wv_vr) # [1, 448, 1024] # Eq.(3) 
        
#         c_i = V_s + v_asta # [1, 448, 1024] # Eq.(4)
        
#         bs,c,hw = c_i.size()
#         spatial_c_i = torch.reshape(c_i.unsqueeze(-1), (bs,c,int(hw**0.5), int(hw**0.5))) #  [1, 448, 32, 32]
        
#         # Similarity-Based Triplet Loss
#         a = wk_vr[0, :, :].detach().clone()
#         b = wk_vr[1:, :, :].detach().clone()
#         wk_vr_neg = torch.cat((b, a.unsqueeze(0)))
#         alpha_negative = F.softmax(torch.matmul(wq_vs, wk_vr_neg) / self.coef, dim=-1)
#         v_negative = torch.matmul(alpha_negative, wv_vr)
        
#         L_tr = F.relu(-v_asta + v_negative + self.gamma).mean()
        
#         return spatial_c_i, L_tr


class Discriminator(nn.Module):
    def __init__(self, nch_in, nch_ker=64, norm='bnorm'):
        super(Discriminator, self).__init__()

        self.nch_in = nch_in
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        # dsc1 : 256 x 256 x 3 -> 128 x 128 x 64
        # dsc2 : 128 x 128 x 64 -> 64 x 64 x 128
        # dsc3 : 64 x 64 x 128 -> 32 x 32 x 256
        # dsc4 : 32 x 32 x 256 -> 16 x 16 x 512
        # dsc5 : 16 x 16 x 512 -> 16 x 16 x 1

        self.dsc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc5 = CNR2d(8 * self.nch_ker, 1,                kernel_size=4, stride=1, padding=1, norm=[],        relu=[], bias=False)

        # self.dsc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=1, padding=1, norm=[], relu=0.2)
        # self.dsc5 = CNR2d(8 * self.nch_ker, 1,                kernel_size=4, stride=1, padding=1, norm=[], relu=[], bias=False)

    def forward(self, x):

        x = self.dsc1(x)
        x = self.dsc2(x)
        x = self.dsc3(x)
        x = self.dsc4(x)
        x = self.dsc5(x)

        # x = torch.sigmoid(x)

        return x


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if gpu_ids:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net




class CBAM(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio, kernel_size):
        super(CBAM, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.channel_attention = ChannelAttention(n_channels_in, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, f):
        chan_att = self.channel_attention(f)
        # print(chan_att.size())
        fp = chan_att * f
        # print(fp.size())
        spat_att = self.spatial_attention(fp)
        # print(spat_att.size())
        fpp = spat_att * fp
        # print(fpp.size())
        return fpp,spat_att

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels = 2, out_channels = 1, kernel_size = kernel_size, padding= int((kernel_size-1)/2))
        # batchnorm
    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim = 1)
        conv = self.conv(pool)
        # batchnorm ????????????????????????????????????????????
        conv = conv.repeat(1,x.size()[1],1,1)
        att = torch.sigmoid(conv)        
        return att
    def agg_channel(self, x, pool = "max"):
        b,c,h,w = x.size()
        x = x.view(b, c, h*w)
        x = x.permute(0,2,1)
        if pool == "max":
            x = F.max_pool1d(x,c)
        elif pool == "avg":
            x = F.avg_pool1d(x,c)
        x = x.permute(0,2,1)
        x = x.view(b,1,h,w)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in/ float(self.reduction_ratio))
        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )
    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel )
        max_pool = F.max_pool2d(x, kernel)
        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)
        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)
        pool_sum = avg_pool_bck + max_pool_bck
        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)
        out = sig_pool.repeat(1,1,kernel[0], kernel[1])
        return out
