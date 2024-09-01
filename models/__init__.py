from .Unet import Unet
from .ILRNet import ILRNet
def unet(params):
    net = Unet(params.kernel_size,params.out_channel,params.out_num)
    net.use_2dconv = False
    net.bandwise = False
    return net

def ilrnet(params):
    net = ILRNet(params)
    net.use_2dconv = False
    net.bandwise = False
    return net
