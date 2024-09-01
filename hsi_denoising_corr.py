import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse

from utility import *
from hsi_setup import Engine, train_options, make_dataset
from utility import dataloaders_hsi_test ###modified###

if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising (Gaussian Noise)')
    opt = train_options(parser)
    print(opt)

    """Setup Engine"""
    engine = Engine(opt)

    """Dataset Setting"""
    HSI2Tensor = partial(HSI2Tensor, use_2dconv=engine.get_net().use_2dconv)

    common_transform_1 = lambda x: x

    common_transform_2 = Compose([
        partial(rand_crop, cropx=32, cropy=32),
    ])

    target_transform = HSI2Tensor()
    # train_transform_0 = Compose([
    #     AddNoise(50),
    #     HSI2Tensor()
    # ])
    train_transform_0 = Compose([
        CorrelatedNoise(),
        HSI2Tensor()
    ])
    train_transform_1 = Compose([
        AddNoiseDynamic(15),
        HSI2Tensor()
    ])
    train_transform_2 = Compose([
        AddNoiseDynamic(55),
        HSI2Tensor()
    ])
    train_transform_3 = Compose([
        AddNoiseDynamic(95),
        HSI2Tensor()
    ])
    # train_transform_3 = Compose([
    #     CorrelatedNoise(),
    #     HSI2Tensor()
    # ])
    train_transform_4 = Compose([
        AddNoiseDynamic(95),
                SequentialSelect(
                    transforms=[
                        lambda x: x,
                        AddNoiseImpulse(),
                        AddNoiseStripe(),
                        AddNoiseDeadline()
                    ]
                ),
        HSI2Tensor()
    ])
    # train_transform_4 = Compose([
    #     AddNoiseDynamicList((15,35,95)),
    #     HSI2Tensor()
    # ])
    '''
    train_transform_1 = Compose([
        AddNoise(50),
        HSI2Tensor()
    ])
    
    train_transform_2 = Compose([
        AddNoiseBlind([10, 30, 50, 70]),
        HSI2Tensor()
    ])
    '''
    print('==> Preparing data..')
    
    icvl_64_31_TL_0 = make_dataset(
        opt, train_transform_0,
        target_transform, common_transform_1,opt.batchSize )
    icvl_64_31_TL_1 = make_dataset(
        opt, train_transform_1,
        target_transform, common_transform_1, opt.batchSize)
    icvl_64_31_TL_2 = make_dataset(
        opt, train_transform_2,
        target_transform, common_transform_1,opt.batchSize)
    icvl_64_31_TL_3 = make_dataset(
        opt, train_transform_3,
        target_transform, common_transform_1, opt.batchSize)
    icvl_64_31_TL_4 = make_dataset(
        opt, train_transform_4,
        target_transform, common_transform_2, opt.batchSize*4)
    icvl_64_31_TL_5 = make_dataset(
        opt, train_transform_4,
        target_transform, common_transform_1, opt.batchSize)
    '''
    icvl_64_31_TL_2 = make_dataset(
        opt, train_transform_2,
        target_transform, common_transform_2, 64)
    '''
    """Test-Dev"""

    ###modified###
    basefolder = opt.testroot
    mat_names = ['icvl_dynamic_512_15','icvl_dynamic_512_55','icvl_dynamic_512_95']

    mat_loaders = []

    if icvl_64_31_TL_0.__len__()*opt.batchSize > 22470:
        max_epoch = 50
        # max_epoch = 100
        if_100 = 0
        epoch_per_save = 10
    else:
        max_epoch = 50
        # max_epoch = 100
        if_100 = 0
        epoch_per_save = 10
    """Main loop"""
    base_lr = opt.lr   
    testsize = 10 ###modified###
    stages=[0, 15, 30, 45, 60, 75]
    while engine.epoch < max_epoch:
        if if_100:
            epoch = engine.epoch * 2
        else:
            epoch = engine.epoch
        display_learning_rate(engine.optimizer)
        np.random.seed() # reset seed per epoch, otherwise the noise will be added with a specific pattern
        if epoch % 20 == 0 and epoch>0 :
            adjust_learning_rate(engine.optimizer, opt.lr*0.5)
            opt.lr = opt.lr*0.5
        engine.train(icvl_64_31_TL_0)
        
        print('\nLatest Result Saving...')
        model_latest_path = os.path.join(engine.basedir, engine.prefix, 'model_latest.pth')
        engine.save_checkpoint(
            model_out_path=model_latest_path
        )

        
        if engine.epoch % epoch_per_save == 0:###modified###
            engine.save_checkpoint()
