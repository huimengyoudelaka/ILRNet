import torch.nn as nn
import torch
import numpy as np
class SADLoss(nn.Module):
    def __init__(self,weight):
        self.weight=weight
        super(SADLoss,self).__init__()
    # def forward(self,X,Y,eps=-13):
    #     X=X.numpy()
    #     Y=Y.numpy()
    #     tmp = (np.sum(X * Y, axis=0) + eps) / ((np.sqrt(np.sum(X ** 2, axis=0)) + eps) * (
    #                 np.sqrt(np.sum(Y ** 2, axis=0)) + eps))
    #     return np.mean(np.real(np.arccos(tmp)))
    def forward(self,X,Y,eps=-13):
        # X=X.numpy()
        # Y=Y.numpy()
        batch_size = X.size()[0]
        tmp = (torch.sum(X * Y, axis=1) + eps) / ((torch.sqrt(torch.sum(X ** 2, axis=1)) + eps) * (
                    torch.sqrt(torch.sum(Y ** 2, axis=1)) + eps))
        return  self.weight*torch.sum(torch.real(torch.acos(tmp)))/batch_size
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        # x : b, c,h,w
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        c_x = x.size()[1]
        count_c = self._tensor_size(x[:, 1:, :, :])
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        c_tv = torch.pow((x[:, 1:, :, :] - x[:, :c_x - 1, : :]), 2).sum()
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w+c_tv/count_c)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        # x : b, c,h,w
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        c_x = x.size()[1]
        count_c = self._tensor_size(x[:, 1:, :, :])
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        c_tv = torch.pow((x[:, 1:, :, :] - x[:, :c_x - 1, : :]), 2).sum()
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w+c_tv/count_c)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
class GDLoss(nn.Module):
    def __init__(self,GDLoss_weight=1,alpha=2):
        super(GDLoss,self).__init__()
        self.GDLoss_weight = GDLoss_weight
        self.alpha=alpha
    def forward(self,x,y):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        c_x = x.size()[1]
        # count_c = self._tensor_size(x[:, 1:, :, :])
        # count_h = self._tensor_size(x[:, :, 1:, :])
        # count_w = self._tensor_size(x[:, :, :, 1:])
        c_tv = torch.pow(torch.abs(x[:, 1:, :, :] - x[:, :c_x - 1, ::])-torch.abs(y[:, 1:, :, :] - y[:, :c_x - 1, ::]), self.alpha).sum()
        h_tv = torch.pow(torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :])-torch.abs(y[:, :, 1:, :] - y[:, :, :h_x - 1, :]), self.alpha).sum()
        w_tv = torch.pow(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1])-torch.abs(y[:,:,:,1:]-y[:,:,:,:w_x-1]), self.alpha).sum()
        return self.GDLoss_weight *(c_tv+h_tv+w_tv)/batch_size
    # def _tensor_size(self, t):
    #     return t.size()[1] * t.size()[2] * t.size()[3]

# def gdl(img_shape, alpha=2):
#     """Image gradient difference loss
#     img_shape: (channels, rows, cols) shape to resize the input
#         vectors, we assume they are input flattened in the spatial dimensions.
#     alpha: l_alpha norm
#     ref: Deep Multi-scale video prediction beyond mean square error,
#          by Mathieu et. al.
#     """
#     def func(y_true, y_pred):
#         y_true = K.batch_flatten(y_true)
#         y_pred = K.batch_flatten(y_pred)
#         Y_true = K.reshape(y_true, (-1, ) + img_shape)
#         Y_pred = K.reshape(y_pred, (-1, ) + img_shape)
#         t1 = K.pow(K.abs(Y_true[:, :, 1:, :] - Y_true[:, :, :-1, :]) -
#                    K.abs(Y_pred[:, :, 1:, :] - Y_pred[:, :, :-1, :]), alpha)
#         t2 = K.pow(K.abs(Y_true[:, :, :, :-1] - Y_true[:, :, :, 1:]) -
#                    K.abs(Y_pred[:, :, :, :-1] - Y_pred[:, :, :, 1:]), alpha)
#         out = K.mean(K.batch_flatten(t1 + t2), -1)
#         return out
#     return func
if __name__ == '__main__':
    x =torch.randn([3,4,5,6])
    y = torch.randn([3, 4, 5, 6])
    sad=SADLoss(0.1)
    z=sad(x,y)
    gdl=GDLoss(0.1,2)
    z=gdl(x,y)
    print(z)
    addition = TVLoss(0.1)
    z = addition(x)
    print(z)