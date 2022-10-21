import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss
from torchvision.models.vgg import *

class ReconstructLoss(nn.Module):
	def __init__(self):
		super(ReconstructLoss, self).__init__()
		self.l1_loss = L1Loss(reduce=True)
		self.l2_loss = MSELoss()
		

	def forward(self, recover_img, gt,defocus_map=None,is_DD=False):
		losses = {}
		if is_DD ==False:
			loss_l1 = self.l1_loss(recover_img[0], gt)
		else:
			loss_l1 = torch.abs(recover_img[0]-gt)
			loss_l1_meanO = loss_l1.mean()
			loss_l1_F = self.l1_loss(recover_img[0], gt)
			deblur, _ = defocus_map.abs().max(dim=1, keepdim=True)
			loss_l1 = loss_l1.mean()+0.02*(deblur*loss_l1).mean()



		losses["total_loss"] = loss_l1

		return losses

class ReconstructLossTest(nn.Module):
    def __init__(self):
        super(ReconstructLossTest, self).__init__()
        self.l1_loss = L1Loss(reduce=True)
        self.l2_loss = MSELoss()

    def forward(self, recover_img, gt):
        losses = {}
        loss_l1 = self.l1_loss(recover_img, gt)
        losses["total_loss"] = loss_l1

        return losses

class PerceptualLoss():
	
	def contentFunc(self):
		conv_3_3_layer = 14
		cnn = vgg19(pretrained=True).features
		cnn = cnn.cuda()
		model = nn.Sequential()
		model = model.cuda()
		for i,layer in enumerate(list(cnn)):
			model.add_module(str(i),layer)
			if i == conv_3_3_layer:
				break
		return model
		
	def __init__(self, loss=MSELoss()):
		self.criterion = loss
		self.contentFunc = self.contentFunc()
			
	def get_loss(self, fakeIm, realIm):
		f_fake = self.contentFunc(fakeIm)
		f_real = self.contentFunc(realIm)
		f_real_no_grad = f_real.detach()
		loss = self.criterion(f_fake, f_real_no_grad)
		return loss