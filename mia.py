import numpy as np
import torch
from torch import nn
from torch.nn.functional import mse_loss, l1_loss
def gradient_img(img):
    device = img.device
    dtype = img.dtype
    # Y = 0.2126 R + 0.7152 G + 0.0722 B
    Y = (0.2126*img[:,0,:,:] + 0.7152*img[:,1,:,:] + 0.0722*img[:,2,:,:]).unsqueeze(1)

    Sx=np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
    kx = torch.from_numpy(Sx).to(device, dtype=dtype, non_blocking=True).unsqueeze(0).unsqueeze(0)
    G_x = nn.functional.conv2d(Y, kx, padding=1)

    Sy=np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
    ky = torch.from_numpy(Sy).to(device, dtype=dtype, non_blocking=True).unsqueeze(0).unsqueeze(0)
    G_y = nn.functional.conv2d(Y, ky, padding=1)

    G=torch.cat((G_x, G_y), dim=1)

    G_mag = (G_x.square() + G_y.square()).sqrt()

    return G , G_mag
def img_gradient_loss(pred, gt, MAX):
    G_pred, G_pred_mag = gradient_img(pred)
    G_gt, G_gt_mag = gradient_img(gt)
    loss = l1_loss(G_pred, G_gt)
    loss_mag = mse_loss(G_pred_mag, G_gt_mag, reduction='none').mean((2,3))
    psnr_per_feature = 10 * torch.log10(MAX**2 / loss_mag)
    psnr_mag = torch.mean(psnr_per_feature)
    return loss, psnr_mag
class ComputeRecLoss:
    def __init__(self, MAX=1.0, w_grad=0.0, compute_grad=True, type='L1', K_ssim=(0.01, 0.4), win_ssim=11):
        self.MAX = MAX
        self.w_grad = w_grad
        self.compute_grad = compute_grad
        self.type = type
        self.K_ssim = K_ssim
        self.win_ssim = win_ssim
        self.size_warn = False
    def __call__(self, T1, T2):
        device = T1.device
        loss_l1, loss_l2, psnr, grad_loss, psnr_mag, loss_msssim = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        loss_l1[0] = l1_loss(T1, T2)
        mse = mse_loss(T1, T2, reduction='none').mean((2,3))
        loss_l2[0] = torch.mean(mse)
        psnr_per_feature = 10 * torch.log10(self.MAX**2 / mse)
        psnr[0] = torch.mean(psnr_per_feature)
        if self.compute_grad:
            grad_loss[0], psnr_mag = img_gradient_loss(T1, T2, self.MAX*4)

        if self.type=='L1':
            loss = loss_l1 + self.w_grad * grad_loss
        elif self.type=='L2':
            loss = loss_l2 + self.w_grad * grad_loss
        elif self.type=='MS-SSIM':
            loss = (1 - loss_msssim) + self.w_grad * grad_loss
        else:
            raise Exception('Supported reconstruction losses are "L1", "L2", "MS-SSIM", but got {}'.format(self.type))

        return loss
