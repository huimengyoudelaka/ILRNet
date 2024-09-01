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
from tqdm import tqdm
import  scipy.io as scio
from ops.utils_blocks import block_module
from skimage.restoration import estimate_sigma
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
    # engine = Engine(opt)
    ###modified###
    basefolder = opt.testroot
    # for noise in (15, 55, 95):
    test_path = os.path.join(basefolder)    
    test = dataloaders_hsi_test.get_dataloaders([test_path], verbose=True, grey=False)

    params = {
        'crop_out_blocks': 0,
        'ponderate_out_blocks': 1,
        'sum_blocks': 0,
        'pad_even': 1,  # otherwise pad with 0 for las
        'centered_pad': 0,  # corner pixel have only one estimate
        'pad_block': 1,  # pad so each pixel has S**2 estimate
        'pad_patch': 0,
        # pad so each pixel from the image has at least S**2 estimate from 1 block
        'no_pad': 0,
        'custom_pad': None,
        'avg': 1}
    block = block_module(opt.patch_size, opt.stride_test, opt.kernel_size, params)
    for batch_idx, (inputs, fname) in enumerate(tqdm(test['test'], disable=True)):
        fname = fname[0]
        print(fname)
        bands = inputs.shape[1]
        num = bands // 31
        output_hsi=torch.zeros_like(inputs)
        for i in range(num + 1):
            start_band = 31 * i
            end_band = 31 * (i + 1)
            if end_band > bands:
                end_band = bands;
                start_band = bands - 31
            split_batch = inputs[:, start_band:end_band, :, :]
            if opt.blind:
                sigma_est = np.array(
                    estimate_sigma(split_batch.squeeze(0).permute([1, 2, 0]).detach().cpu(), multichannel=True,
                                   average_sigmas=False)).max() * 255
            else:
                sigma_est = opt.noise_level
            print(sigma_est)
            if sigma_est > 55:
                opt.resumePath=('/nas_data/xiongfc/MTSNMF/checkpoints/mtsnmf_u/cave_mtsnmf0_0.0005_K9_F196_U64_L5_95_All/model_latest.pth')
            if sigma_est > 15 and sigma_est <= 55:
                opt.resumePath=('/nas_data/xiongfc/MTSNMF/checkpoints/mtsnmf_u/cave_mtsnmf0_0.0005_K9_F196_U64_L5_55_All/model_latest.pth')
            if sigma_est <=15:
                opt.resumePath = (
                    '/nas_data/xiongfc/MTSNMF/checkpoints/mtsnmf_u/cave_mtsnmf0_0.0005_K9_F196_U64_L5_15_All/model_latest.pth')
            engine = Engine(opt)
            outputs=engine.test_real(split_batch, block, opt.batchSize, savedir = None)
            output_hsi[:, start_band:end_band, :, :] = outputs
        scio.savemat(fname + 'Res.mat', {'output': output_hsi.cpu().detach().numpy()})







    # _, _, res_arr = engine.validate(, '')
    # scio.savemat(str(noise) + 'dB_' + 'Res.mat',
    #              {'res_arr': res_arr})    

    # for noise in (15,55,95):
    #     test_path = os.path.join(basefolder, str(noise)+'dB/')
    #     print('noise:   ',noise,end='')
    #     test = dataloaders_hsi_test.get_dataloaders([test_path],verbose=True,grey=False)
    #
    #     # res_arr, input_arr = engine.test_develop(mat_loader, savedir=resdir, verbose=True)
    #     # print(res_arr.mean(axis=0))
    #     _,_,res_arr=engine.validate(test['test'], '')

    ###modified###
