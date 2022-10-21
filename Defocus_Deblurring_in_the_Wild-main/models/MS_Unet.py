# 多尺度去模糊
import torch.nn as nn
from models.rcab import RCAB, ResidualGroup, default_conv
import math
import torch
import torch.nn.functional as F
from models.unet_parts import *


def kernel_conv(kernel_size, input_dim, reduction, max_pool=False, upsample=False):
    res_conv = []
    out_dim = input_dim
    if max_pool ==True :
       out_dim = input_dim*2
    if upsample ==True :
       out_dim = input_dim/2
    
    if kernel_size <= 1:
        res_conv = [nn.Conv2d(input_dim, input_dim, kernel_size=1, stride=1, padding=0), nn.ReLU(True)]
        return nn.Sequential(*res_conv)  
    elif kernel_size ==2:
        res_conv = [
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(True),
            ]
        
    else:
        res_conv.append(ResidualGroup(default_conv, input_dim, 3, reduction, n_resblocks=math.floor(kernel_size/3)))

    if max_pool:
        res_conv.append(nn.MaxPool2d(kernel_size=2, stride=2))

    if upsample:
        res_conv.append(nn.Upsample(scale_factor=2))

    return nn.Sequential(*res_conv)


def connect_conv(input_dim, output_dim, kernel_size, stride, padding, bias=True, dilation=1):
    conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=bias, dilation=dilation)
    relu = nn.ReLU(True)

    return nn.Sequential(*[conv, relu])


class KernelEDNet(nn.Module):
    def __init__(self):
        super(KernelEDNet, self).__init__()
        kernel_size = [1,4,7,10] #本来是[1,4,7,10]
        self.kernel_size = kernel_size
        self.channel = 64
        self.head = connect_conv(3, self.channel, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        convk_tail = nn.Conv2d(self.channel * 1, 3, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        reluk_tail = nn.Sigmoid()
        self.tail_hard = nn.Sequential(*[convk_tail, reluk_tail])
        
        self.connect = nn.Conv2d(self.channel * 1, self.channel, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        self.connect2 = nn.Conv2d(self.channel * 1, self.channel, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        
        self.unetdown4 = Down4(self.channel,self.channel)        
        self.unetdown2 = Down(self.channel,self.channel)
        self.unetup2 = Up(self.channel,self.channel,True)
        self.unetup4 = Up4(self.channel,self.channel)



        self.layer1 = nn.ModuleList()
        for k in kernel_size:
            self.layer1.append(kernel_conv(k, self.channel, 16, max_pool=True)) #256
 
        self.layer2 = nn.ModuleList()
        for k in kernel_size:
            self.layer2.append(kernel_conv(k, self.channel, 16, max_pool=True))  #128
 
        self.layer3 = nn.ModuleList()
        for k in kernel_size:
            self.layer3.append(kernel_conv(k,self.channel, 16, max_pool=True))  #64 

        self.layer4 = nn.ModuleList()
        for k in kernel_size:
            self.layer4.append(kernel_conv(k, self.channel, 16, max_pool=True))  #32

        self.layer5 = nn.ModuleList()
        for k in kernel_size:
            self.layer5.append(kernel_conv(k, self.channel, 16, upsample=True))  #64

        self.layer6 = nn.ModuleList()
        for k in kernel_size:
            self.layer6.append(kernel_conv(k, self.channel, 16, upsample=True))   #128

        self.layer7 = nn.ModuleList()
        for k in kernel_size:
            self.layer7.append(kernel_conv(k, self.channel, 16, upsample=True))   #256

        self.layer8 = nn.ModuleList()
        for k in kernel_size:
            self.layer8.append(kernel_conv(k, self.channel, 16, upsample=True))    #512


        self.MAX_TRAINNUM = 2e4
        self.iter_num = 1
    
    def forward(self, x, gt=None):
        blur, _ = x[:, 3:, :, :].abs().max(dim=1, keepdim=True)
        x = x[:, :3, :, :]
        x = self.head(x)
        blur_mask = []
        blur_mask_d1=[]
        blur_mask_d2=[]
        
        blur_d1 = F.interpolate(blur,scale_factor=0.25)
        blur_d2 = F.interpolate(blur,scale_factor=0.125)

        feature_layer = []
        feature_layer_d4 = []
        feature_layer_d8 = []

        if gt is not None:
            self.iter_num += 1

        static_kernel_size =  [0.0,1.9,4.2,6.2]#[0.0,1.8,4.5,6.9]
        for kernel_bound, kernel_up in zip(static_kernel_size, static_kernel_size[1:]):
            mask = ((blur >= kernel_bound) & (blur < kernel_up)).float()
            blur_mask.append(mask)

        mask = (blur >= static_kernel_size[-1]).float()
        blur_mask.append(mask)
        #下采样4倍掩膜
        static_kernel_size_d =  [0.0,1.9,4.2,6.2] #[0,4,12]
        for kernel_bound, kernel_up in zip(static_kernel_size_d, static_kernel_size_d[1:]):
            mask = ((blur_d1 >= kernel_bound) & (blur_d1 < kernel_up)).float()
            blur_mask_d1.append(mask)

        mask = (blur_d1 >= static_kernel_size_d[-1]).float()
        blur_mask_d1.append(mask)
        #下采样8倍掩膜
        static_kernel_size_d =  [0.0,1.9,4.2,6.2] #[0,4,12]
        for kernel_bound, kernel_up in zip(static_kernel_size_d, static_kernel_size_d[1:]):
            mask = ((blur_d2 >= kernel_bound) & (blur_d2 < kernel_up)).float()
            blur_mask_d2.append(mask)

        mask = (blur_d2 >= static_kernel_size_d[-1]).float()
        blur_mask_d2.append(mask)

        ######################
        # #x为原分辨率
        layer_output1 = []
        for i in range(len(self.kernel_size)):
            layer_output1.append(self.layer1[i](x))                    #256

        layer_output2 = []
        for i in range(len(self.kernel_size)):
            res_x = F.adaptive_avg_pool2d(x, layer_output1[i].size()[2:])
            layer_output2.append(self.layer2[i](res_x+layer_output1[i]))           #128

        layer_output3 = []
        for i in range(len(self.kernel_size)):
            res_x = F.adaptive_avg_pool2d(x, layer_output2[i].size()[2:])
            layer_output3.append(self.layer3[i](res_x+layer_output2[i]))             #64
        
        layer_output4 = []
        for i in range(len(self.kernel_size)):
            res_x = F.adaptive_avg_pool2d(x, layer_output3[i].size()[2:])             #32
            layer_output4.append(self.layer4[i](res_x+layer_output3[i]))

        layer_output5 = []
        for i in range(len(self.kernel_size)):
            res_x = F.adaptive_avg_pool2d(x, layer_output4[i].size()[2:])
            layer_output5.append(self.layer5[i](res_x + layer_output4[i]))           #64


        layer_output6 = []
        for i in range(len(self.kernel_size)):
            res_x = F.adaptive_avg_pool2d(x, layer_output3[i].size()[2:])
            layer_output6.append((self.layer6[i](res_x+layer_output5[i] )))     #128

        layer_output7 = []
        for i in range(len(self.kernel_size)):
            res_x = F.adaptive_avg_pool2d(x, layer_output6[i].size()[2:])
            layer_output7.append((self.layer7[i](res_x+layer_output6[i] + layer_output2[i])))     #256

        layer_outputx = []
        for i in range(len(self.kernel_size)):
            res_x = F.adaptive_avg_pool2d(x, layer_output7[i].size()[2:])
            layer_outputx.append(self.layer8[i](res_x+layer_output7[i] + layer_output1[i]))       #512
        
        if self.iter_num < self.MAX_TRAINNUM:
            iter_weight = torch.exp(torch.tensor(- (self.iter_num * 2 / self.MAX_TRAINNUM) ** 2))
            for layer_i, blur_i in zip(layer_outputx, blur_mask):
                feature_layer.append((layer_i * blur_i * iter_weight + (1-iter_weight) * layer_i).unsqueeze(0))
            
        else:
            feature_layer = [layer_i.unsqueeze(0) for layer_i in layer_outputx]
        tempx = torch.cat(feature_layer, dim=0).sum(dim=0)
        ########################
        ##x下采样4倍
        xd4=self.unetdown4(tempx)
        layer_output1 = []
        for i in range(len(self.kernel_size)):
            layer_output1.append(self.layer1[i](xd4))                    #256/dx

        layer_output2 = []
        for i in range(len(self.kernel_size)):
            res_x = F.adaptive_avg_pool2d(xd4, layer_output1[i].size()[2:])
            layer_output2.append(self.layer2[i](res_x+layer_output1[i]))           #128/dx

        
        layer_output7 = []
        for i in range(len(self.kernel_size)):
            res_x = F.adaptive_avg_pool2d(xd4, layer_output2[i].size()[2:])
            layer_output7.append((self.layer7[i](res_x + layer_output2[i])))     #256/dx

        layer_outputxd4 = []
        for i in range(len(self.kernel_size)):
            res_x = F.adaptive_avg_pool2d(xd4, layer_output7[i].size()[2:])
            layer_outputxd4.append(self.layer8[i](res_x+layer_output7[i] + layer_output1[i]))       #512/dx
            
        if self.iter_num < self.MAX_TRAINNUM:
            iter_weight = torch.exp(torch.tensor(- (self.iter_num * 2 / self.MAX_TRAINNUM) ** 2))            
            for layer_i, blur_i in zip(layer_outputxd4, blur_mask_d1):    
                feature_layer_d4.append((layer_i * blur_i * iter_weight + (1-iter_weight) * layer_i).unsqueeze(0))            
        else:           
            feature_layer_d4 = [layer_i.unsqueeze(0) for layer_i in layer_outputxd4]
        tempxd4=torch.cat(feature_layer_d4, dim=0).sum(dim=0)
        ##########################
        ###下采样到1/8
        xd8=self.unetdown2(tempxd4)
        layer_output1 = []
        for i in range(len(self.kernel_size)):
            layer_output1.append(self.layer1[i](xd8))                    #256/dx

        layer_outputxd8 = []
        for i in range(len(self.kernel_size)):
            res_x = F.adaptive_avg_pool2d(xd8, layer_output1[i].size()[2:])
            layer_outputxd8.append((self.layer7[i](res_x + layer_output1[i])))     #256/dx


        if self.iter_num < self.MAX_TRAINNUM:
            iter_weight = torch.exp(torch.tensor(- (self.iter_num * 2 / self.MAX_TRAINNUM) ** 2))
            for layer_i, blur_i in zip(layer_outputxd8, blur_mask_d2):    
                feature_layer_d8.append((layer_i * blur_i * iter_weight + (1-iter_weight) * layer_i).unsqueeze(0))
        else:
            feature_layer_d8 = [layer_i.unsqueeze(0) for layer_i in layer_outputxd8]
        tempxd8=torch.cat(feature_layer_d8, dim=0).sum(dim=0)+xd8
        xup2 = self.unetup2(tempxd8,xd4)
        xup4 = self.unetup4(tempxd4+xup2)
                
        x = x + xup4+tempx
        out = self.tail_hard(x)
        
        return [out]