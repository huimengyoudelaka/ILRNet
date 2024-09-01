# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from utility import dataloaders_hsi_test
from utility import *
from hsi_setup import Engine, train_options
import models
import  scipy.io as scio
from ops.utils_blocks import block_module
import numpy as np

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


prefix = 'test'

if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising')
    opt = train_options(parser)
    print(opt)

    cuda = not opt.no_cuda
    opt.no_log = True

    """Setup Engine"""
    engine = Engine(opt)
    ###modified###
    basefolder = opt.testroot
    noise=opt.noise_level
    test_path = os.path.join(basefolder, str(noise))
    print('noise:   ', noise, end='')
    test = dataloaders_hsi_test.get_dataloaders([test_path], verbose=True, grey=False)
    _, _, res_arr = engine.validate(test['test'], '')
    from showRes import saveAsCsv
    aver = [np.mean(res_arr, axis=0)]
    saveName = opt.resumePath.split('/')
    savePath = './res/'+str(noise)+'/'+saveName[-2]
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    savePath = savePath +'/'+saveName[-1].split('.')[0]+'_'
    saveAsCsv(savePath+str(noise) + 'dB_' + 'Res.csv',res_arr)
    saveAsCsv(savePath+str(noise) + 'dB_' + 'ResAver.csv',aver)