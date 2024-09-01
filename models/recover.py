import torch
import torch.nn as nn
from .testSvd import svdv2_1
# import pywt
# import pytorch_wavelet as wavelet
from pytorch_wavelets import DWTForward, DWTInverse
import  scipy.io as scio

class recoverBlock(nn.Module):
    def __init__(self):
        super(recoverBlock, self).__init__()
        self.thre1=nn.Parameter(torch.zeros(1))

    def RX(self,X,shape):
        b,channel,c,h,w=shape
        X_0=torch.reshape(X,[b,channel,c,h*w])
        X_0=X_0.permute(0,1,3,2)
        return X_0

    def RTX(self,X,shape):
        b,channel,c,h,w=shape

        X_0=X.permute(0,1,3,2)
        X_0 = torch.reshape(X_0, [b, channel, c, h, w])
        return X_0

    def recoverFromSvd(self,M):
        U,S,V=svdv2_1.apply(M)
        VT=V.permute(0,1,3,2)
        torch.save(S.cpu(), "/data_3/yejin/experiment_mat/irdnet_remake/lr_visual/S_wavelet.pt")
        mythre = torch.sigmoid(self.thre1) * S[:,:,0]
        # torch.save(self.thre1.cpu(), "/data_3/yejin/experiment_mat/irdnet_remake/lr_visual/thre1.pt")
        mythre=torch.unsqueeze(mythre,-1)
        S=S-mythre
        S=torch.relu(S)
        # torch.save(S.cpu(), "/data_3/yejin/experiment_mat/irdnet_remake/lr_visual/S_after.pt")
        S = torch.diag_embed(S)
        US=torch.matmul(U,S)
        USV=torch.matmul(US,VT)

        # mask = torch.zeros(31).to('cuda:0')
        # temp_res = []
        # for i in range(5):
        #     mask = mask * 0
        #     mask[i] = 1
        #     temp_S = S * mask
        #     temp_S = torch.diag_embed(temp_S)
        #     US=torch.matmul(U,temp_S)
        #     USV=torch.matmul(US,VT)
        #     temp_res.append(USV)

        # S = torch.diag_embed(S)
        # US=torch.matmul(U,S)
        # USV=torch.matmul(US,VT)

        return USV

    # def DWT(self,M):
    #     M_shape = M.shape
    #     M = torch.reshape(M, [-1,M_shape[-3],M_shape[-2],M_shape[-1]])
    #     xfm = DWTForward(J=1, mode='zero', wave='haar').to(M.device)
    #     ifm = DWTInverse(mode='zero', wave='haar').to(M.device)
    #     Ml, Mh = xfm(M)
    #     Ml = torch.reshape(Ml, list(M_shape[:-2])+[M_shape[-2]//2,M_shape[-1]//2])
    #     Ml_shape = Ml.shape
    #     x = self.RX(Ml,Ml_shape)
    #     lr_x = self.recoverFromSvd(x)
    #     lr_Ml = self.RTX(lr_x,Ml_shape)
    #     lr_Ml = torch.reshape(lr_Ml, [-1,lr_Ml.shape[-3],lr_Ml.shape[-2],lr_Ml.shape[-1]])
    #     recoverM = ifm((lr_Ml, Mh))
    #     recoverM = torch.reshape(recoverM, M_shape)
    #     return recoverM

    def DWT(self,M):
        M_shape = M.shape

        # scio.savemat('/data_3/yejin/experiment_mat/irdnet_remake/lr_visual/pre.mat', {'ilrnet': M.cpu().detach().numpy()})

        M = torch.reshape(M, [-1,M_shape[-3],M_shape[-2],M_shape[-1]])
        xfm = DWTForward(J=1, mode='zero', wave='haar').to(M.device)
        ifm = DWTInverse(mode='zero', wave='haar').to(M.device)
        Ml, Mh = xfm(M)
        Ml = torch.reshape(Ml, list(M_shape[:-2])+[M_shape[-2]//2,M_shape[-1]//2])
        lr_Ml = self.lr_approximate(Ml)
        # for index ,lr_Ml_item in enumerate(lr_Ml_list):
        #     lr_Ml_item = torch.reshape(lr_Ml_item, [-1,lr_Ml.shape[-3],lr_Ml.shape[-2],lr_Ml.shape[-1]])
        #     recoverM = ifm((lr_Ml_item, Mh))
        #     recoverM = torch.reshape(recoverM, M_shape)
        #     scio.savemat('/data_3/yejin/experiment_mat/irdnet_remake/lr_visual/pre'+str(index)+'.mat', {'ilrnet': recoverM.cpu().detach().numpy()})
        lr_Ml = torch.reshape(lr_Ml, [-1,lr_Ml.shape[-3],lr_Ml.shape[-2],lr_Ml.shape[-1]])
        recoverM = ifm((lr_Ml, Mh))
        recoverM = torch.reshape(recoverM, M_shape)

        # scio.savemat('/data_3/yejin/experiment_mat/irdnet_remake/lr_visual/after.mat', {'ilrnet': recoverM.cpu().detach().numpy()})

        return recoverM

    def lr_approximate(self,Ml):
        Ml_shape = Ml.shape
        x = self.RX(Ml,Ml_shape)
        lr_x = self.recoverFromSvd(x)
        # lr_x, lr_list = self.recoverFromSvd(x)
        # temp = []
        # for lr in lr_list:
        #     temp.append(self.RTX(lr,Ml_shape))
        lr_Ml = self.RTX(lr_x,Ml_shape)
        return lr_Ml


    def forward(self, x):
        # return self.lr_approximate(x)
        # temp = x.view(*x.shape[:-2],-1)
        # U, S, V = torch.svd(temp,some=True,compute_uv=True)
        # torch.save(S.cpu(), "/data_3/yejin/experiment_mat/irdnet_remake/lr_visual/S_pre_.pt")
        # res = self.DWT(x)
        # temp = res.view(*x.shape[:-2],-1)
        # U, S, V = torch.svd(temp,some=True,compute_uv=True)
        # torch.save(S.cpu(), "/data_3/yejin/experiment_mat/irdnet_remake/lr_visual/S_after_.pt")
        # return res
        return self.DWT(x)
