import torch
import torch.optim as optim
import models
from math import inf
from tqdm import tqdm
import  scipy.io as scio
import os
import argparse
from ops.utils_blocks import block_module
from os.path import join
from utility import *
from utility.ssim import SSIMLoss
from utility import dataloaders_hsi_test
import time

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1/len(self.losses)] * len(self.losses)
    
    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            total_loss += loss(predict, target) * weight
        return total_loss

    def extra_repr(self):
        return 'weight={}'.format(self.weight)


def train_options(parser):
    def _parse_str_args(args):
        str_args = args.split(',')
        parsed_args = []
        for str_arg in str_args:
            arg = int(str_arg)
            if arg >= 0:
                parsed_args.append(arg)
        return parsed_args    
    parser.add_argument('--prefix', '-p', type=str, default='sru3d_nobn_test',
                        help='prefix')
    parser.add_argument('--arch', '-a', metavar='ARCH',
                        default='mscnet',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names))
    parser.add_argument('--batchSize', '-b', type=int,
                        default=1, help='training batch size. default=16')         
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate. default=1e-3.')
    parser.add_argument('--wd', type=float, default=0,
                        help='weight decay. default=0')
    parser.add_argument('--loss', type=str, default='l2',
                        help='which loss to choose.', choices=['l1', 'l2', 'smooth_l1', 'ssim', 'l2_ssim'])
    parser.add_argument('--init', type=str, default='kn',
                        help='which init scheme to choose.', choices=['kn', 'ku', 'xn', 'xu', 'edsr'])
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda?')
    parser.add_argument('--no-log', action='store_true',
                        help='disable logger?')
    parser.add_argument('--threads', type=int, default=8,
                        help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=2018,
                        help='random seed to use. default=2018')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--no-ropt', '-nro', action='store_true',
                            help='not resume optimizer')          
    parser.add_argument('--chop', action='store_true',
                            help='forward chop')                                      
    parser.add_argument('--resumePath', '-rp', type=str,
                        default=None, help='checkpoint to use.')
    parser.add_argument('--dataroot', '-d', type=str,
                        default='datasets/ICVL64_31.db', help='data root')
    parser.add_argument('--clip', type=float, default=1e6)
    parser.add_argument('--gpu-ids', type=str, default='1,3', help='gpu ids')
    parser.add_argument('--noise_level', type=str, default=15)
    parser.add_argument('--blind', type=int, default=0)



    parser.add_argument('--outdir', type=str, default='/home/xiongfc/tesla_data/MTCSC_Fu/')
    ###MSCNet
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--out_channel', type=int, default=16)
    parser.add_argument('--grow', type=int, default=16)
    parser.add_argument('--grow_num', type=int, default=3)
    parser.add_argument('--out_num', type=int, default=5)
    parser.add_argument('--unfoldings', type=int, default=5)
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--verbose', type=bool, default=0)
    parser.add_argument('--testroot', '-tr', type=str,default= '/home/fugym/HDD/fugym/ICVL/test')
    parser.add_argument('--gtroot', '-gr', type=str,default= '/home/fugym/HDD/fugym/ICVL/test/test_crop/')
    opt = parser.parse_args()
    opt.gpu_ids = _parse_str_args(opt.gpu_ids)
    return opt


def make_dataset(opt, train_transform, target_transform, common_transform, batch_size=None, repeat=1):
    dataset = LMDBDataset(opt.dataroot, repeat=repeat)
    # dataset.length -= 1000
    # dataset.length = size or dataset.length

    """Split patches dataset into training, validation parts"""
    dataset = TransformDataset(dataset, common_transform)

    train_dataset = ImageTransformDataset(dataset, train_transform, target_transform)

    #train_loader = DataLoader(train_dataset,
    #                          batch_size=batch_size or opt.batchSize, shuffle=True,
    #                          num_workers=opt.threads, pin_memory=not opt.no_cuda, worker_init_fn=worker_init_fn)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size or opt.batchSize, shuffle=True,
                              num_workers=opt.threads, pin_memory=not opt.no_cuda, worker_init_fn=worker_init_fn)
    return train_loader


class Engine(object):
    def __init__(self, opt):
        self.prefix = opt.prefix
        self.opt = opt
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.basedir = None
        self.iteration = None
        self.epoch = None
        self.best_psnr = None
        self.best_loss = None
        self.writer = None

        self.__setup()

    def __setup(self):
        self.basedir = join(self.opt.outdir, 'checkpoints', self.opt.arch)
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)

        self.best_psnr = 0
        self.best_loss = 1e6
        self.epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.iteration = 0
        '''
        cuda_list = str(self.opt.gpu_ids)[1:-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_list
        gpus_list = []
        for gpus in range(len(self.opt.gpu_ids)):
            gpus_list.append(gpus)
        self.opt.gpu_ids = gpus_list
        '''
        cuda = not self.opt.no_cuda
        self.device = 'cuda:'+str(self.opt.gpu_ids[0]) if cuda else 'cpu'
        print('Cuda Acess: %d' % cuda)
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        torch.manual_seed(self.opt.seed)
        if cuda:
            torch.cuda.manual_seed(self.opt.seed)

        """Model"""
        print("=> creating model '{}'".format(self.opt.arch))
        # from models.Unet import ListaParams
        from models.ILRNet import ListaParams

        # params = ListaParams(kernel_size=self.opt.kernel_size, out_channel=self.opt.out_channel,
        #                     layer=self.opt.layer)
        params = ListaParams(kernel_size=self.opt.kernel_size, out_channel=self.opt.out_channel, grow=self.opt.grow,
                         grow_num=self.opt.grow_num, out_num=self.opt.out_num,
                         unfoldings=self.opt.unfoldings)
        #from models.IRDNet import ListaParams
        
        #params = ListaParams(kernel_size=self.opt.kernel_size, out_channel=self.opt.out_channel,grow=self.opt.grow, unfoldings=self.opt.unfoldings)
        with torch.cuda.device(self.opt.gpu_ids[0]):
            self.net = models.__dict__[self.opt.arch](params)
        # initialize parameters
        
        init_params(self.net, init_type=self.opt.init) # disable for default initialization

        if len(self.opt.gpu_ids) > 1:
            from models.sync_batchnorm import DataParallelWithCallback
            self.net = DataParallelWithCallback(self.net, device_ids=self.opt.gpu_ids)
        
        if self.opt.loss == 'l2':
            self.criterion = nn.MSELoss()
        if self.opt.loss == 'l1':
            self.criterion = nn.L1Loss()
        if self.opt.loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        if self.opt.loss == 'ssim':
            self.criterion = SSIMLoss(data_range=1, channel=31)
        if self.opt.loss == 'l2_ssim':
            self.criterion = MultipleLoss([nn.MSELoss(), SSIMLoss(data_range=1, channel=31)], weight=[1, 2.5e-3])
        
        print(self.criterion)

        if cuda:
            self.net.to(self.device)
            self.criterion = self.criterion.to(self.device)

        """Logger Setup"""
        log = not self.opt.no_log
        if log:
            self.writer = get_summary_writer(os.path.join(self.basedir, 'logs'), self.opt.prefix)

        """Optimization Setup"""
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.opt.lr, weight_decay=self.opt.wd, amsgrad=False)
        
        """Resume previous model"""
        if self.opt.resume:
            # Load checkpoint.
            self.load(self.opt.resumePath, not self.opt.no_ropt)
            self.optimizer.param_groups[0]['capturable'] = True
        else:
            print('==> Building model..')
            print(self.net)

    def forward(self, inputs):        
        if self.opt.chop:            
            output = self.forward_chop(inputs)
        else:
            output = self.net(inputs)
        
        return output

    def forward_chop(self, x, base=16):        
        n, c, b, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        
        shave_h = np.ceil(h_half / base) * base - h_half
        shave_w = np.ceil(w_half / base) * base - w_half

        shave_h = shave_h if shave_h >= 10 else shave_h + base
        shave_w = shave_w if shave_w >= 10 else shave_w + base

        h_size, w_size = int(h_half + shave_h), int(w_half + shave_w)        
        
        inputs = [
            x[..., 0:h_size, 0:w_size],
            x[..., 0:h_size, (w - w_size):w],
            x[..., (h - h_size):h, 0:w_size],
            x[..., (h - h_size):h, (w - w_size):w]
        ]

        outputs = [self.net(input_i) for input_i in inputs]

        output = torch.zeros_like(x)
        output_w = torch.zeros_like(x)
        
        output[..., 0:h_half, 0:w_half] += outputs[0][..., 0:h_half, 0:w_half]
        output_w[..., 0:h_half, 0:w_half] += 1
        output[..., 0:h_half, w_half:w] += outputs[1][..., 0:h_half, (w_size - w + w_half):w_size]
        output_w[..., 0:h_half, w_half:w] += 1
        output[..., h_half:h, 0:w_half] += outputs[2][..., (h_size - h + h_half):h_size, 0:w_half]
        output_w[..., h_half:h, 0:w_half] += 1
        output[..., h_half:h, w_half:w] += outputs[3][..., (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
        output_w[..., h_half:h, w_half:w] += 1
        
        output /= output_w

        return output

    def __step(self, train, inputs, targets):        
        if train:
            self.optimizer.zero_grad()
        loss_data = 0
        total_norm = None
        if self.get_net().bandwise:
            O = []
            for time_, (i, t) in enumerate(zip(inputs.split(1, 1), targets.split(1, 1))):
                o = self.net(i)
                O.append(o)
                loss = self.criterion(o, t)
                if train:
                    loss.backward()
                loss_data += loss.item()
            outputs, res = torch.cat(O, dim=1)
        else:
            outputs = self.net(inputs)
            for index,output in enumerate(outputs):
                loss = self.criterion(output, targets)
                if(index<(len(outputs)-1)):
                    loss = loss*0.5
                loss_data += loss
            
            if train:
                # loss.backward()
                loss_data.backward()
            # loss_data += loss.item()
        if train:
            total_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.opt.clip)
            self.optimizer.step()
        return outputs, loss_data, total_norm

    def load(self, resumePath=None, load_opt=True):
        model_best_path = join(self.basedir, self.prefix, 'model_latest.pth')
        if os.path.exists(model_best_path):
            # best_model = torch.load(model_best_path,map_location='cuda:0')
            best_model = torch.load(model_best_path,map_location='cuda:'+str(self.opt.gpu_ids[0]))

        print('==> Resuming from checkpoint %s..' % resumePath)
        assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resumePath or model_best_path,map_location=torch.device(self.device))
        #### comment when using memnet
        self.epoch = checkpoint['epoch'] 
        self.iteration = checkpoint['iteration']
        if load_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        ####
        pytorch_total_params = sum(p.numel() for p in  self.get_net().parameters())
        # for p in self.get_net().parameters():
        #     print(p.name)
        print('Nb tensors: ', len(list(self.get_net().named_parameters())), "; Trainable Params: ", pytorch_total_params)
        self.get_net().load_state_dict(checkpoint['net'])
        # self.print_state_dict(self.get_net().named_parameters())
    def print_state_dict(self, state_dict):
        for name, param in state_dict:
            name = name.split('.')
            print(name[0],'\t',param.shape)
            scio.savemat(name[0]+'.mat',{str(name[0]):param.cpu().detach().numpy()})

        # for layer in state_dict:
        #     print('\t', state_dict[layer].shape)
        #     break

    def train(self, train_loader):
        print('\nEpoch: %d' % self.epoch)
        self.net.train()
        train_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if not self.opt.no_cuda:
                inputs, targets = inputs.to(self.device), targets.to(self.device)            
            outputs, loss_data, total_norm = self.__step(True, inputs, targets)
            train_loss += loss_data
            avg_loss = train_loss / (batch_idx+1)

            if not self.opt.no_log:
                self.writer.add_scalar(
                    join(self.prefix, 'train_loss'), loss_data, self.iteration)
                self.writer.add_scalar(
                    join(self.prefix, 'train_avg_loss'), avg_loss, self.iteration)

            self.iteration += 1

            progress_bar(batch_idx, len(train_loader), 'AvgLoss: %.4e | Loss: %.4e | Norm: %.4e' 
                         % (avg_loss, loss_data, total_norm))

        self.epoch += 1
        if not self.opt.no_log:
            self.writer.add_scalar(
                join(self.prefix, 'train_loss_epoch'), avg_loss, self.epoch)


    def validate(self, valid_loader, name):
        self.net.eval()
        validate_loss = 0
        total_psnr = 0
        total_ssim = 0
        total_sam = 0
        print('\n[i] Eval dataset {}...'.format(name))
        # print(torch.cuda.device_count())
        torch.cuda.empty_cache()
        res_arr = np.zeros((len(valid_loader), 3))
        # filenames = []
        with torch.no_grad():
            with torch.cuda.device(self.opt.gpu_ids[0]):
                for batch_idx, (inputs, fname) in enumerate(tqdm(valid_loader, disable=True)):
                    fname = fname[0]
                    targets = dataloaders_hsi_test.get_gt(self.opt.gtroot, fname)
                    inputs = inputs.unsqueeze(0)
                    targets = targets.unsqueeze(0).unsqueeze(0)
                    inputs, targets = inputs[:, :, :, :, :], targets[:, :, :, :, :]
                    if not self.opt.no_cuda:
                        self.device = 'cuda:' + str(self.opt.gpu_ids[0]) if not self.opt.no_cuda else 'cpu'
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs, loss_data, _ = self.__step(False, inputs, targets)
                    psnr = np.mean(cal_bwpsnr(outputs[-1], targets))
                    res_arr[batch_idx, :] = MSIQA(outputs[-1], targets)
                    validate_loss += loss_data
                    avg_loss = validate_loss / (batch_idx + 1)
                    total_psnr += psnr
                    avg_psnr = total_psnr / (batch_idx + 1)
                    avg_sam = total_sam / (batch_idx + 1)
                    # avg_time = total_time/(batch_idx + 1)
                    # avg_lamdas = total_lamdas/(batch_idx + 1)
                    # progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f | SSIM: %.4f |SAM: %.4f'
                    #                           % (avg_loss, avg_psnr, avg_ssim, avg_sam))
                    progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f'
                                 % (avg_loss, avg_psnr))
                # if batch_idx == 10:###modified###
                #    break###modified###
        if not self.opt.no_log:
            self.writer.add_scalar(
                join(self.prefix, name, 'val_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(
                join(self.prefix, name, 'val_psnr_epoch'), avg_psnr, self.epoch)
        return avg_psnr, avg_loss, res_arr

    def validate_patch(self, valid_loader, name):
        self.net.eval()
        validate_loss = 0
        total_psnr = 0
        total_ssim = 0
        total_sam = 0
        print('\n[i] Eval dataset {}...'.format(name))
        # print(torch.cuda.device_count())
        torch.cuda.empty_cache()
        res_arr = np.zeros((len(valid_loader), 3))
        with torch.no_grad():
            with torch.cuda.device(self.opt.gpu_ids[0]):
                for batch_idx, (inputs, fname) in enumerate(tqdm(valid_loader, disable=True)):
                    fname = fname[0]
                    targets = dataloaders_hsi_test.get_gt(self.opt.gtroot, fname)
                    kernel_size = (inputs.shape[-3],256,256)
                    stride =(155,44,44)
                    self.device = 'cuda'
                    col_data,data_shape = read_HSI(inputs[0].cpu().numpy(),kernel_size=kernel_size,stride=stride,device=self.device)  
                    inputs = col_data
                    outputs = torch.empty_like(inputs).to(inputs.device)

                    if not self.opt.no_cuda:
                        self.device = 'cuda:' + str(self.opt.gpu_ids[0]) if not self.opt.no_cuda else 'cpu'
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs, loss_data, _ = self.__step(False, inputs, inputs)
                    outputs = refold(outputs[-1],data_shape=data_shape, kernel_size=kernel_size,stride=stride,device=self.device).unsqueeze(0).unsqueeze(0)

                    psnr = np.mean(cal_bwpsnr(outputs, targets))
                    res_arr[batch_idx, :] = MSIQA(outputs[-1], targets)
                    validate_loss += loss_data
                    avg_loss = validate_loss / (batch_idx + 1)
                    total_psnr += psnr
                    avg_psnr = total_psnr / (batch_idx + 1)
                    # avg_ssim =total_ssim/(batch_idx+1)
                    avg_sam = total_sam / (batch_idx + 1)
                    # progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f | SSIM: %.4f |SAM: %.4f'
                    #                           % (avg_loss, avg_psnr, avg_ssim, avg_sam))
                    progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f'
                                 % (avg_loss, avg_psnr))
                # if batch_idx == 10:###modified###
                #    break###modified###
        if not self.opt.no_log:
            self.writer.add_scalar(
                join(self.prefix, name, 'val_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(
                join(self.prefix, name, 'val_psnr_epoch'), avg_psnr, self.epoch)

        # np.savetxt("avglamdas.csv",avg_lamdas.numpy()) 
        # np.savetxt("avglamdas1.csv",avg_lamdas1.numpy()) 
        # np.savetxt("avglamdas2.csv",avg_lamdas2.numpy()) 
        return avg_psnr, avg_loss, res_arr


    def save_checkpoint(self, model_out_path=None, **kwargs):
        if not model_out_path:
            model_out_path = join(self.basedir, self.prefix, "model_epoch_%d_%d.pth" % (
                self.epoch, self.iteration))

        state = {
            'net': self.get_net().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'iteration': self.iteration,
        }
        
        state.update(kwargs)

        if not os.path.isdir(join(self.basedir, self.prefix)):
            os.makedirs(join(self.basedir, self.prefix))

        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    # saving result into disk
    def test_develop(self, test_loader, savedir=None, verbose=True):
        from scipy.io import savemat
        from os.path import basename, exists

        def torch2numpy(hsi):
            if self.net.use_2dconv:
                R_hsi = hsi.data[0].cpu().numpy().transpose((1,2,0))
            else:
                R_hsi = hsi.data[0].cpu().numpy()[0,...].transpose((1,2,0))
            return R_hsi    

        self.net.eval()
        test_loss = 0
        total_psnr = 0
        dataset = test_loader.dataset.dataset

        res_arr = np.zeros((len(test_loader), 3))
        input_arr = np.zeros((len(test_loader), 3))

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if not self.opt.no_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs, loss_data, _ = self.__step(False, inputs, targets)
                
                test_loss += loss_data
                avg_loss = test_loss / (batch_idx+1)
                
                res_arr[batch_idx, :] = MSIQA(outputs, targets)
                input_arr[batch_idx, :] = MSIQA(inputs, targets)

                """Visualization"""
                # Visualize3D(inputs.data[0].cpu().numpy())
                # Visualize3D(outputs.data[0].cpu().numpy())

                psnr = res_arr[batch_idx, 0]
                ssim = res_arr[batch_idx, 1]
                if verbose:
                    print(batch_idx, psnr, ssim)

                if savedir:
                    filedir = join(savedir, basename(dataset.filenames[batch_idx]).split('.')[0])  
                    outpath = join(filedir, '{}.mat'.format(self.opt.arch))

                    if not exists(filedir):
                        os.mkdir(filedir)

                    if not exists(outpath):
                        savemat(outpath, {'R_hsi': torch2numpy(outputs)})
                        
        return res_arr, input_arr

    def test_real(self, inputs, block, batch_size, savedir=None):
        self.net.eval()
        # print(torch.cuda.device_count())
        # torch.cuda.empty_cache()
        with torch.no_grad():
            self.device = 'cuda:' + str(self.opt.gpu_ids[0]) if not self.opt.no_cuda else 'cpu'
            inputs = inputs.to(self.device)
            batch_noisy_blocks = block._make_blocks(inputs)
            patch_loader = torch.utils.data.DataLoader(batch_noisy_blocks, batch_size=batch_size,
                                                       drop_last=False)
            batch_out_blocks = torch.zeros_like(batch_noisy_blocks)
            # batch_share_blocks = torch.zeros_like(batch_noisy_blocks)
            #batch_unique_blocks = torch.zeros_like(batch_noisy_blocks)
            for i, inp in enumerate(patch_loader):  # if it doesnt fit in memory
                id_from, id_to = i * patch_loader.batch_size, (i + 1) * patch_loader.batch_size
                # _,batch_share_blocks[id_from:id_to],_= self.net(
                #     inp)
                batch_out_blocks[id_from:id_to]= self.net(
                    inp)
            outputs = block._agregate_blocks(batch_out_blocks)
            #unique = block._agregate_blocks(batch_unique_blocks)
            # share = block._agregate_blocks(batch_share_blocks)
            #scio.savemat('AnaRes_unique.mat', {'avg': unique.cpu().detach().numpy()})

            # scio.savemat('AnaRes.mat', {'share': share.cpu().detach().numpy(),'unique': unique.cpu().detach().numpy(),'avg': avg.cpu().detach().numpy(),'output': outputs.cpu().detach().numpy()})
        return outputs

    def get_net(self):
        if len(self.opt.gpu_ids) > 1:
            return self.net.module
        else:
            return self.net           
