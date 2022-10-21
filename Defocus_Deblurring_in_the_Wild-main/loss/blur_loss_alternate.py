from tokenize import Single
import torch
import torch.nn as nn
from torch.nn import L1Loss
from kornia.filters import  gaussian_blur2d
import torch.nn.functional as F
from kornia.losses import TotalVariation
import numpy as np
from loss.generate_bw_kernel import bw_kernel_generator

def generate_vectorc(radius):
    abs_radius = radius.abs()
    basic_disk = torch.ones(abs_radius * 2 + 1, abs_radius * 2 + 1)
    for i in range(abs_radius * 2 + 1):
        for j in range(abs_radius * 2 + 1):
            if ((i-abs_radius) ** 2 + (j-abs_radius) ** 2) > abs_radius ** 2:
                basic_disk[i][j] = 0.0
    sign_radius = radius.sign()
    blur_kernel = torch.zeros_like(basic_disk)
    for i in range(2 * abs_radius):
        center_point = i * sign_radius + abs_radius
        start_point = max(center_point - abs_radius, 0)
        end_point = min(center_point + abs_radius + 1, abs_radius * 2+1)
        blur_kernel[:,start_point:end_point] += basic_disk[:, start_point:end_point] ** 2

    sum_kernel = blur_kernel.sum()
    blur_kernel = blur_kernel/(sum_kernel)
    return blur_kernel

#The Butterworth high pass filter is constructed. 
# First, the estimated fuzzy kernel is used to replace the Gaussian fuzzy kernel, 
# and then the unsupervised fuzzy kernel is estimated. 
# If necessary, parallax estimation can be used to estimate COC, 
# and the patch like method can be used for supervision
class GemorestructNet(nn.Module):
    def __init__(self):
        super(GemorestructNet, self).__init__()
        # MSELoss()
        self.beta = 0.1
        self.gamma = 1
        self.radius = 25
        self.radius_set=self.radius_dictA(self.radius)
        self.weight_pos_set =nn.ModuleList()
        for i in range(0, self.radius):
            if i==0:
                k=0
            elif (i-2)%7==0:
                k=k+2
            else:
                k=k+1
            if k==0:
                current_conv =nn.Conv2d(1, 1, kernel_size=k * 2 + 1,padding=0, bias=False)
            else:
                current_conv = nn.Conv2d(1, 1, kernel_size=k * 2 + 1, bias=False)  
            current_conv.weight.data = self.radius_set[i].unsqueeze(0).unsqueeze(0)
            current_conv.requires_grad_(True)
            self.weight_pos_set.append(current_conv)
        self.weight_neg_set = nn.ModuleList()
        for i in range(0,self.radius):
            if i==0:
                k=0
            elif (i-2)%7==0:
                k=k+2
            else:
                k=k+1
            if (i-2)%7==0:
                k=k+2
            else:
                k=k+1
            if k==0:
                current_conv =nn.Conv2d(1, 1, kernel_size=k * 2 + 1,padding=0, bias=False)
            else:
                current_conv = nn.Conv2d(1, 1, kernel_size=k * 2 + 1, bias=False)  
            current_conv.weight.data = self.radius_set[i].unsqueeze(0).unsqueeze(0)
            current_conv.requires_grad_(True)
            self.weight_neg_set.append(current_conv)
        #self.radius_set, self.weight_pos_set, self.weight_neg_set = self.radius_dictA(self.radius)

    def radius_dictA(self, c_radius):
        radius_set = [torch.tensor([[1.]])]
        bw_para_list=[]
        for order in [3, 6, 9]:
            for cut_off_factor in [2.5,2]:
                for beta in [0.1, 0.2]:
                    bw_para_list.append([order,cut_off_factor,beta])
        smooth_strength=7 # this is realted to kappa in the main paper as kappa = 1/smooth_strength
        for i in range(1,c_radius):
            order,cut_off_factor,beta = bw_para_list[ 0 %len(bw_para_list)]
            kernel_c, kernel_r, kernel_l = bw_kernel_generator(2*abs(i)+1, order, cut_off_factor, beta, smooth_strength)
            radius_set.append(torch.tensor(kernel_c).float()) 
        return radius_set

    def radius_dictA2(self, radius_set,c_radius):
        weight_pos_set = []
        weight_neg_set = []
        k=0
        for i in range(0, c_radius):
            if i==0:
                k=0
            elif (i-2)%7==0:
                k=k+2
            else:
                k=k+1
            if k==0:
                current_conv =nn.Conv2d(1, 1, kernel_size=k * 2 + 1,padding=0, bias=False)
            else:
                current_conv = nn.Conv2d(1, 1, kernel_size=k * 2 + 1, bias=False)  
            current_conv.weight.data = radius_set[i].unsqueeze(0).unsqueeze(0)
            current_conv.requires_grad_(True)
            weight_pos_set.append(current_conv.cuda())
        k=0
        for i in range(0, c_radius ):
            if i==0:
                k=0
            elif (i-2)%7==0:
                k=k+2
            else:
                k=k+1
            if (i-2)%7==0:
                k=k+2
            else:
                k=k+1
            if k==0:
                current_conv =nn.Conv2d(1, 1, kernel_size=k * 2 + 1,padding=0, bias=False)
            else:
                current_conv = nn.Conv2d(1, 1, kernel_size=k * 2 + 1, bias=False)  
            current_conv.weight.data = radius_set[i].unsqueeze(0).unsqueeze(0)
            current_conv.requires_grad_(True)
            weight_neg_set.append(current_conv.cuda())
        k=0

        return radius_set, weight_pos_set, weight_neg_set 

    def compute_gemoS(self, blur_map, x):
        SingleDP_gemo = torch.zeros_like(blur_map)
        k=0
        for i in range(self.radius):
            if i ==0:
               k=0
            elif (i-2)%7==0:
                k=k+2
            else:
                k=k+1
            current_left = F.pad(x[:, :3, :, :], pad=(k, k, k, k)) 

            lpos_mask = ((i - 1 < blur_map) & (blur_map <= i)).float()
            rpos_mask = ((i < blur_map) & (blur_map < i + 1)).float()
            lneg_mask = ((-(i + 1) < blur_map) & (blur_map <= -i)).float()
            rneg_mask = ((-i < blur_map) & (blur_map < -(i - 1))).float()

            pos_mask = (blur_map - i + 1) * lpos_mask + (i + 1 - blur_map) * rpos_mask
            neg_mask = (blur_map + i + 1) * lneg_mask + (-blur_map - i + 1) * rneg_mask 
            
            if i == 0:
                pos_mask = pos_mask * 0.5
                neg_mask = neg_mask * 0.5

            if (pos_mask.sum() + neg_mask.sum()) > 0:
                for j in range(3):
                    SingleDP_gemo[:, j:j+1, :, :] += \
                        self.weight_pos_set[i](current_left[:, j:j+1, :, :]) * pos_mask[:, j:j+1, :, :]
                    SingleDP_gemo[:, j:j+1, :, :] += \
                        self.weight_neg_set[i](current_left[:, j:j+1, :, :]) * neg_mask[:, j:j+1, :, :]
        k=0

        return SingleDP_gemo


    def forward(self, blur_map, SingleDPAoF):

        SingleDP_gemo = self.compute_gemoS(blur_map, SingleDPAoF)
        
        return SingleDP_gemo


class GemoLoss(nn.Module):
    def __init__(self):
        super(GemoLoss, self).__init__()
        self.mse_loss = L1Loss()
        # MSELoss()
        self.alpha = {
            "unsurpervise": 1000,
            "tv_loss": 1e-5,
        }
        self.beta = 0.1
        self.gamma = 1
        self.epoch = 0
        self.tv_loss = TotalVariation()


    def blur_mean_loss(self, img1, img2):
        diff = (img1 - img2).abs()
        #diff = gaussian_blur2d(diff, (3, 3), (1.5, 1.5)).abs()
        return diff.mean()

    def forward(self, blur_map, SingleDP_gemo,SingleDPOoF):
        losses = {}
        unsurpervise_loss = self.blur_mean_loss(SingleDP_gemo, SingleDPOoF) * self.alpha["unsurpervise"]
        losses["unsurpervise loss"] = unsurpervise_loss
        tv_loss = self.tv_loss(blur_map).sum() * self.alpha["tv_loss"]
        losses["tv loss"] = tv_loss
        if self.epoch < 3000:
            losses["total_loss"] = unsurpervise_loss
        else:
            losses["total_loss"] = unsurpervise_loss + tv_loss
        self.epoch += 1
        return losses
    



    




