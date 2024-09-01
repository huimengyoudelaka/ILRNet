import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
import torch.nn.functional as F
ListaParams = namedtuple('ListaParams', ['kernel_size', 'out_channel','grow','grow_num','out_num','unfoldings'])
from  models.Unet import Unet
import math
import torch.fft as fft
class ILRNet(nn.Module):
    def __init__(self, params: ListaParams):
        super(ILRNet, self).__init__()
        self.init=Unet(kernel_size=3, out_channel=params.out_channel,
                         layer=params.out_num)
        self.refs=nn.ModuleList()
        for i in range(params.unfoldings):
            self.refs.append(Unet(kernel_size=3, out_channel=params.grow,
                         layer=params.grow_num))
        self.unfoldings = params.unfoldings
        self.w1=(WeighEstimator())
        self.w2=(WeighEstimator())
    def forward(self, I):
        X =self.init(I) ### initial
        res=[]
        ind=0
        tempW1 = []
        tempW2 = []
        for index,ref in enumerate(self.refs):
            weight=self.w1(torch.cat([X,I],dim=1))
            temp_X = I*(1-weight)+weight*X
            refined_X = ref(temp_X)
            weight=self.w2(torch.cat([X,temp_X],dim=1))
            X=weight*X+(1-weight)*refined_X
        res.append(X)
        return res

class SELayer(nn.Module):
    def __init__(self, channel=1, sup=16):
        super(SELayer, self).__init__()
        # self.avg_pool = torch.nn.AvgPool2d(3,1,1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel*sup, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel*sup, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, bands, h, w = x.size()
        x = x.permute([0,2,3,4,1])
        g = self.fc(x).permute([0,4,1,2,3])
        return g

class WeighEstimator(nn.Module):
    def __init__(self):
        super(WeighEstimator, self).__init__()
        # self.time_emb = PositionalEmbedding(16)
        # self.activate = torch.nn.functional.relu
        # self.bias = nn.Linear(16, 1)
        self.beta_estimator = nn.Sequential(
            # layer1
            nn.Conv2d(
                in_channels=2, out_channels=64, kernel_size=5, stride=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # layer2
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # layer3
            nn.Conv2d(
                in_channels=128, out_channels=1, kernel_size=3, stride=1
            ),
            nn.Sigmoid(),
        )
    def forward(self, I):
        # print(I.shape)
        # I=I.squeeze(1)
        bs,_, c, h, w = I.shape
        # I_nlm =I.cpu().numpy()
        # sigma_est = estimate_sigma(I_nlm, multichannel=True, average_sigmas=False)
        block = min(56, h)
        c_w = (w - block) // 2
        c_h = (h - block) // 2
        # time_tensor = torch.ones(bs * c).to(I.device)*time
        to_estimate = I[:,:,:, c_h: c_h + block, c_w: c_w + block].permute([0,2,1,3,4]).contiguous().view(
            bs * c, _, block, block
        )
        # to_estimate += self.bias(self.activate(self.time_emb(time_tensor)))[:,:,None,None]
        # print(to_estimate.shape)
        beta = self.beta_estimator(to_estimate)
        beta = beta.view(bs,1, c, 1, 1)
        return beta

class WeighEstimator1(nn.Module):
    def __init__(self):
        super(WeighEstimator1, self).__init__()
        # self.time_emb = PositionalEmbedding(16)
        # self.activate = torch.nn.functional.relu
        # self.bias = nn.Linear(16, 1)
        self.beta_estimator = nn.Sequential(
            # layer1
            nn.Conv2d(
                in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            # layer2
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            # layer3
            nn.Conv2d(
                in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1
            ),
            nn.Sigmoid(),
        )
    def forward(self, I):
        # print(I.shape)
        # I=I.squeeze(1)
        bs,_, c, h, w = I.shape
        I = I.permute([0,2,1,3,4])
        I = I.view(bs*c,2,h,w)
        beta = self.beta_estimator(I)
        beta = beta.view(bs,1, c, h, w)
        return beta

class WeighEstimator2(nn.Module):
    def __init__(self):
        super(WeighEstimator2, self).__init__()
        # self.time_emb = PositionalEmbedding(16)
        # self.activate = torch.nn.functional.relu
        # self.bias = nn.Linear(16, 1)
        self.beta_estimator = nn.Sequential(
            # layer1
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=5, stride=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # layer2
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # layer3
            nn.Conv2d(
                in_channels=128, out_channels=1, kernel_size=3, stride=1
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid(),
        )
    def forward(self, I):
        bs,_, c, h, w = I.shape
        I = I.view(bs*c,1,h,w)
        beta = self.beta_estimator(I)
        beta = beta.view(bs,1, c, 1, 1)
        return beta

class WeighEstimator3(nn.Module):
    def __init__(self):
        super(WeighEstimator3, self).__init__()
        # self.time_emb = PositionalEmbedding(16)
        # self.activate = torch.nn.functional.relu
        # self.bias = nn.Linear(16, 1)
        self.beta_estimator = nn.Sequential(
            # layer1
            nn.Conv2d(
                in_channels=31, out_channels=64, kernel_size=5, stride=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # layer2
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # layer3
            nn.Conv2d(
                in_channels=128, out_channels=31, kernel_size=3, stride=1
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid(),
        )
    def forward(self, I):
        bs,_, c, h, w = I.shape
        I = I.view(bs,c,h,w)
        beta = self.beta_estimator(I)
        beta = beta.view(bs,1, c, 1, 1)
        return beta

class Down(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=False):
        super(Down, self).__init__()
        self.add_module('conv', nn.Conv3d(in_channels, channels, k, s, p, bias=False))
        # self.add_module('bn', BatchNorm3d(channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        # self.add_module('mta', MTAtt(channels=channels))

class WeighEstimator4(nn.Module):
    def __init__(self):
        super(WeighEstimator4, self).__init__()
        # self.time_emb = PositionalEmbedding(16)
        # self.activate = torch.nn.functional.relu
        # self.bias = nn.Linear(16, 1)
        self.beta_estimator = nn.Sequential(
            # layer1
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=5, stride=2
            ),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # layer2
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2
            ),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # layer3
            nn.Conv2d(
                in_channels=128, out_channels=1, kernel_size=3, stride=1
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid(),
        )
    def forward(self, I):
        bs,_, c, h, w = I.shape
        I = I.view(bs*c,1,h,w)
        beta = self.beta_estimator(I)
        beta = beta.view(bs,1, c, 1, 1)
        return beta

class SPPLayer(nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = h // (2 ** i)
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x

class WeighEstimator5(nn.Module):
    def __init__(self,spp_level=2):
        super(WeighEstimator5, self).__init__()
        self.spp_level = spp_level
        self.num_grids = 0
        for i in range(spp_level):
            self.num_grids += 2**(i*2)
        self.beta_estimator = nn.Sequential(
            # layer1
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=5, stride=2
            ),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # layer2
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1
            ),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # layer3
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1
            ),
            nn.ReLU(),
            # layer4
            SPPLayer(self.spp_level),
            nn.Linear(self.num_grids*128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )
    def forward(self, I):
        bs,_, c, h, w = I.shape
        I = I.view(bs*c,1,h,w)
        beta = self.beta_estimator(I)
        beta = beta.view(bs,1, c, 1, 1)
        return beta

class WeighEstimator6(nn.Module):
    def __init__(self,spp_level=3):
        super(WeighEstimator6, self).__init__()
        self.spp_level = spp_level
        self.num_grids = 0
        for i in range(spp_level):
            self.num_grids += 2**(i*2)
        self.beta_estimator = nn.Sequential(
            # layer1
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=5, stride=3
            ),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # layer2
            # nn.Conv2d(
            #     in_channels=64, out_channels=128, kernel_size=3, stride=2
            # ),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # layer3
            SPPLayer(self.spp_level),
            nn.Linear(self.num_grids*64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )
    def forward(self, I):
        bs,_, c, h, w = I.shape
        I = I.view(bs*c,1,h,w)
        beta = self.beta_estimator(I)
        beta = beta.view(bs,1, c, 1, 1)
        return beta

class PositionalEmbedding(nn.Module):
    __doc__ = r"""Computes a positional embedding of timesteps.

    Input:
        x:?tensor of shape (N)
    Output:
        tensor of shape (N, dim)
    Args:
        dim (int): embedding dimension
        scale (float): linear scale to be applied to timesteps. Default: 1.0
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb    

class WeighEstimator7(nn.Module):
    def __init__(self):
        super(WeighEstimator7, self).__init__()
        # self.time_emb = PositionalEmbedding(16)
        # self.activate = torch.nn.functional.relu
        # self.bias = nn.Linear(16, 1)
        self.beta_estimator = nn.Sequential(
            # layer1
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=5, stride=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # layer2
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # layer3
            nn.Conv2d(
                in_channels=128, out_channels=1, kernel_size=3, stride=1
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid(),
        )
    def forward(self, I1, I2):
        bs,_, c, h, w = I1.shape
        I2 = torch.flip(I2,[-2,-1])
        I1_fft = torch.fft.fft2(I1, dim=(-2, -1), norm='ortho')
        I2_fft = torch.fft.fft2(I2, dim=(-2, -1), norm='ortho')
        I = I1_fft * I2_fft
        I = torch.fft.irfft2(I, s=(h,w), dim=(-2, -1), norm='ortho')
        I = I.view(bs*c,1,h,w)
        beta = self.beta_estimator(I)
        beta = beta.view(bs,1, c, 1, 1)
        return beta

class WeighEstimator8(nn.Module):
    def __init__(self, in_dim=1, out_dim=8, patch = 2):
        super(WeighEstimator8, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.img2token1 = nn.Conv2d(
                in_channels=in_dim, out_channels=out_dim, kernel_size=patch, stride=patch
        )
        self.img2token2 = nn.Conv2d(
                in_channels=in_dim, out_channels=out_dim, kernel_size=patch, stride=patch
        )
        self.query = nn.Linear(out_dim, out_dim, bias=False)
        self.key = nn.Linear(out_dim, out_dim, bias=False)
        self.value = nn.Linear(out_dim, out_dim, bias=False)
        self.beta_estimator = nn.Sequential(
            # layer1
            nn.Conv2d(
                in_channels=8, out_channels=64, kernel_size=5, stride=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # layer3
            nn.Conv2d(
                in_channels=64, out_channels=1, kernel_size=3, stride=1
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        bs,_, c, h, w = x.shape
        x = x.view(bs*c,1,h,w)
        y = y.view(bs*c,1,h,w)
        x = self.img2token1(x)
        y = self.img2token2(y)
        token_h = x.shape[2]
        token_w = x.shape[3]
        x = x.permute([0,2,3,1]).contiguous().view(bs*c,x.shape[2]*x.shape[3],self.out_dim)
        y = y.permute([0,2,3,1]).contiguous().view(bs*c,y.shape[2]*y.shape[3],self.out_dim)
        x = self.query(x)
        y = self.key(y)
        attn_scores = torch.matmul(x, y.transpose(-2, -1)) / (self.out_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        V = self.value(y)
        output = torch.bmm(attn_weights, V)
        output = output.permute([0,2,1]).contiguous().view(bs*c,self.out_dim,token_h,token_w)
        # to_estimate += self.bias(self.activate(self.time_emb(time_tensor)))[:,:,None,None]
        # print(to_estimate.shape)
        beta = self.beta_estimator(output)
        beta = beta.view(bs,1, c, 1, 1)
        return beta

def filter(img):
    shape = img.shape
    f = fft.fftshift(img,dim=(-2,-1))
    mask = torch.ones(f.shape[-2],f.shape[-1])
    indexes1 = torch.arange(0-mask.shape[-1]//2, mask.shape[-1] -mask.shape[-1]//2).to(img.device)
    indexes1, indexes2 = torch.meshgrid(indexes1, indexes1, indexing='ij')
    mask = (torch.sigmoid(indexes1**2 + indexes2**2)-0.5)*2
    f = f * mask
    res = fft.ifftshift(f,dim=(-2,-1))
    return res

class FFTfilter(nn.Module):
    def __init__(self):
        super(FFTfilter, self).__init__()
        self.para = torch.nn.Parameter(torch.tensor([5.0]))

    def forward(self, img):
        shape = img.shape
        f = fft.fftshift(img,dim=(-2,-1))
        mask = torch.ones(f.shape[-2],f.shape[-1])
        indexes1 = torch.arange(0-mask.shape[-1]//2, mask.shape[-1] -mask.shape[-1]//2).to(img.device)
        indexes1, indexes2 = torch.meshgrid(indexes1, indexes1, indexing='ij')
        mask = 1 - torch.exp(-(indexes1**2 + indexes2**2)/(2*(self.para**2)))
        f = f * mask
        res = fft.ifftshift(f,dim=(-2,-1))
        return res


class LayerNorm3D(nn.Module):
    def __init__(self, eps=1e-6):
        super(LayerNorm3D, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones((1,)))
        self.beta = torch.nn.Parameter(torch.zeros((1,)))
        self.eps = eps

    def forward(self, x):
        B, C, Bands, H, W = x.shape
        mean = x.reshape(B, -1).mean(-1).reshape(B, 1, 1, 1, 1)
        std = x.reshape(B, -1).std(-1).reshape(B, 1, 1, 1, 1)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

def norm(I):
    B, C, Bands, H, W = I.shape
    x = I.reshape(B, -1)
    maxV = torch.max(x,dim=1)[0]
    maxV = maxV.reshape(B, 1, 1, 1, 1)
    minV = torch.min(x,dim=1)[0]
    minV = minV.reshape(B, 1, 1, 1, 1)
    return (I-minV)/(1e-6+maxV-minV)


if __name__ == '__main__':
    x = torch.randn([2, 1, 31, 64, 64])
    params = ListaParams(kernel_size=3, out_channel=16, grow=16,
                         grow_num=3, out_num=5,
                         unfoldings=5)
    u=IRDNet18(params)
    u(x)


