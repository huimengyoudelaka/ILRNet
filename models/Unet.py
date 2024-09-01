import torch.nn.functional
from collections import namedtuple
import torch.nn as nn
from .recover import recoverBlock

ListaParams = namedtuple('ListaParams', ['kernel_size', 'out_channel', 'layer'])
from models.sync_batchnorm import SynchronizedBatchNorm2d, SynchronizedBatchNorm3d

BatchNorm3d = SynchronizedBatchNorm3d
BatchNorm2d = SynchronizedBatchNorm2d


class Unet(nn.Module):
    def __init__(self, kernel_size=3, out_channel=16,
                         layer=5):
        in_channels = 1
        num_half_layer = layer
        channels = out_channel
        super(Unet, self).__init__()
        # Encoder
        # assert downsample is None or 0 < downsample <= num_half_layer
        interval = 2
        self.feature_extractor = Conv3dBNReLU(in_channels, channels)
        self.encoder = nn.ModuleList()
        # self.gate_encoder=nn.ModuleList()
        for i in range(0, num_half_layer):
            if i % interval:
                encoder_layer = Conv3dBNReLU(channels, channels)
            else:
                encoder_layer = Down(channels, 2 * channels, k=kernel_size, s=(1, 2, 2), p=1)
                channels *= 2
            self.encoder.append(encoder_layer)
            # self.gate_encoder.append(MTAtt(channels=channels))
        # Decoder
        self.decoder = nn.ModuleList()
        # self.gate_decoder = nn.ModuleList()
        if num_half_layer % 2 != 0:
            for i in range(0, num_half_layer):
                if i % interval:
                    if i==1 and num_half_layer >= 5:
                    # if i==1:
                        decoder_layer = DeConv3dBNReLU_rocover(channels, channels)
                    else:
                        decoder_layer = DeConv3dBNReLU(channels, channels)
                    # decoder_layer = DeConv3dBNReLU_rocover(channels, channels)
                    # decoder_layer = DeConv3dBNReLU(channels, channels)
                    
                else:
                    decoder_layer = UpsampleConv3dBNReLU(channels, channels // 2)
                    channels //= 2
                self.decoder.append(decoder_layer)
            # self.gate_decoder.append(MTAtt(channels=channels))
        if num_half_layer % 2 == 0:
            for i in range(0, num_half_layer):
                if i % interval:
                    # if i==1 and num_half_layer >= 5:
                    # # if i==1:
                    #     decoder_layer = DeConv3dBNReLU_rocover(channels, channels)
                    # else:
                    #     decoder_layer = DeConv3dBNReLU(channels, channels)
                    # decoder_layer = DeConv3dBNReLU_rocover(channels, channels)
                    decoder_layer = UpsampleConv3dBNReLU(channels, channels // 2)
                    channels //= 2
                else:
                    decoder_layer = DeConv3dBNReLU(channels, channels)
                self.decoder.append(decoder_layer)
        self.reconstructor = DeConv3dReLU(channels, in_channels)
        # self.enl_1 = EfficientNL(in_channels=channels)

    # = None, head_count = None,  = None
    def forward(self, x):
        num_half_layer = len(self.encoder)
        xs = [x]
        out = self.feature_extractor(xs[0])
        xs.append(out)
        if num_half_layer % 2 != 0:
            for i in range(num_half_layer - 1):
                # out = (self.encoder[i](out))
                out = self.encoder[i](out)
                xs.append(out)
            # out = (self.encoder[-1](out))
            out = self.encoder[-1](out)
            # out = self.decoder[0](out)
            out = self.decoder[0](out)

            for i in range(1, num_half_layer):
                out = out + xs.pop()
                out = self.decoder[i](out)
                # out = self.decoder[i](out)
            out = (out) + xs.pop()
            out = self.reconstructor(out)
            out = (out) + xs.pop()
        else:
            # out = self.encoder[0](out)
            # out = self.gates[0](self.encoder[0](out))
            for i in range(0, num_half_layer):
                out = self.encoder[i](out)
                # out = self.gates[i](self.encoder[i](out))
                xs.append(out)
                # print(i,out.shape)
                # print(out.shape)
            # out = self.encoder[-1](out)
            # out = self.nl_1(out)
            # out = self.decoder[0](out)
            for i in range(0, num_half_layer):
                # print(i,out.shape)
                out = out + xs.pop()
                out = self.decoder[i](out)
            # print(out.shape)
            out = (out) + xs.pop()
            out = self.reconstructor(out)
            # print(out.shape)
            temp = xs.pop()
            # print(temp.shape)
            out = (out) + temp
        return out




class Conv3dReLU(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=False, bn=False):
        super(Conv3dReLU, self).__init__()
        self.add_module('conv', nn.Conv3d(in_channels, channels, k, s, p, bias=False))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('conv2', nn.Conv3d(channels, channels, k, s, p, bias=False))
        self.add_module('relu2', nn.ReLU(inplace=inplace))


class DeConv3dReLU(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=False, bn=False):
        super(DeConv3dReLU, self).__init__()
        self.add_module('deconv2', nn.ConvTranspose3d(in_channels, in_channels, k, s, p, bias=False))
        self.add_module('deconv', nn.ConvTranspose3d(in_channels, channels, k, s, p, bias=False))

        # self.add_module('relu', nn.ReLU(inplace=inplace))
        # self.add_module('deconv2', nn.ConvTranspose3d(channels, channels, k, s, p, bias=False))
        # self.add_module('relu2', nn.ReLU(inplace=inplace))


class UpsampleConv3dBNReLU(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, upsample=(1, 2, 2), inplace=False):
        super(UpsampleConv3dBNReLU, self).__init__()
        self.add_module('upsample_conv', UpsampleConv3d(in_channels, channels, k, s, p, bias=False, upsample=upsample))
        # self.add_module('bn', BatchNorm3d(channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        # self.add_module('mta', MTAtt(channels=channels))



class DeConv3dBNReLU(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=False):
        super(DeConv3dBNReLU, self).__init__()
        self.add_module('deconv', nn.ConvTranspose3d(in_channels, in_channels, k, s, p, bias=False))
        # self.add_module('bn', BatchNorm3d(channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))

        # self.add_module('recover1',recoverBlock())

        self.add_module('deconv2', nn.ConvTranspose3d(in_channels, channels, k, s, p, bias=False))
        # self.add_module('bn2', BatchNorm3d(channels))
        self.add_module('relu2', nn.ReLU(inplace=inplace))

class DeConv3dBNReLU_rocover(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=False):
        super(DeConv3dBNReLU_rocover, self).__init__()
        self.add_module('deconv', nn.ConvTranspose3d(in_channels, in_channels, k, s, p, bias=False))
        # self.add_module('bn', BatchNorm3d(channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))

        self.add_module('recover1',recoverBlock())

        self.add_module('deconv2', nn.ConvTranspose3d(in_channels, channels, k, s, p, bias=False))
        # self.add_module('bn2', BatchNorm3d(channels))
        self.add_module('relu2', nn.ReLU(inplace=inplace))


class Conv3dBNReLU(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=False):
        super(Conv3dBNReLU, self).__init__()
        self.add_module('conv', nn.Conv3d(in_channels, channels, k, s, p, bias=False))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('conv2', nn.Conv3d(channels, channels, k, s, p, bias=False))
        self.add_module('relu2', nn.ReLU(inplace=inplace))


class Down(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=False):
        super(Down, self).__init__()
        self.add_module('conv', nn.Conv3d(in_channels, channels, k, s, p, bias=False))
        # self.add_module('bn', BatchNorm3d(channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        # self.add_module('mta', MTAtt(channels=channels))



class UpsampleConv3d(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, upsample=None):
        super(UpsampleConv3d, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample, mode='trilinear', align_corners=True)

        self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        # self.mta=MTAtt(channels=out_channels)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.conv3d(x_in)

        return out


if __name__ == '__main__':
    x = torch.randn([2, 1, 31, 64, 64])
    params = ListaParams(kernel_size=3, out_channel=16,
                         layer=5)
    u=Unet(kernel_size=3, out_channel=16,
                         layer=5)
    u(x)


