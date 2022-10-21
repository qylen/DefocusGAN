import torch
import torch.nn as nn

import functools

import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd


import torch.nn.functional as F

from utils.util import ImagePool
import copy

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_D(input_nc, ndf, which_model_netD, n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[],
             use_parallel=True):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids, use_parallel=use_parallel)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids, use_parallel=use_parallel)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False, gpu_ids=[],
                 use_parallel=True):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input).cuda()

class GANLoss(nn.Module):
    def __init__(
            self, use_l1=True, target_real_label=1.0,
            target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_l1:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor
    
    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real).cuda()
        return self.loss(input, target_tensor)


class DiscLoss(nn.Module):
    def name(self):
        return 'DiscLoss'

    def __init__(self):
        super(DiscLoss, self).__init__()

        self.criterionGAN = GANLoss(use_l1=False)
        #self.fake_AB_pool = ImagePool(50)

    def get_g_loss(self, net, fakeB, realB):
        # First, G(A) should fake the discriminator
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake, 1)

    def get_loss(self, net, fakeB, realB):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        self.pred_fake = net(fakeB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, 0)

        # Real
        self.pred_real = net(realB)
        self.loss_D_real = self.criterionGAN(self.pred_real, 1)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def __call__(self, net, fakeB, realB):
        return self.get_loss(net, fakeB, realB)

class DiscLossLS(DiscLoss):
    def name(self):
        return 'DiscLossLS'

    def __init__(self):
        super(DiscLossLS, self).__init__()
        self.criterionGAN = GANLoss(use_l1=True)

    def get_g_loss(self, net, fakeB, realB):
        return DiscLoss.get_g_loss(self, net, fakeB)

    def get_loss(self, net, fakeB, realB):
        return DiscLoss.get_loss(self, net, fakeB, realB)

class DiscLossWGANGP(DiscLossLS):
    def name(self):
        return 'DiscLossWGAN-GP'

    def __init__(self):
        super(DiscLossWGANGP, self).__init__()
        self.LAMBDA = 10

    def get_g_loss(self, net, fakeB, realB,defocus_map):
        # First, G(A) should fake the discriminator
        # defocus_map=defocus_map.detach()
        # deblur, _ = defocus_map.abs().max(dim=1, keepdim=True)
        self.D_fake = net(fakeB)
        # _,_,imageH,imageW = self.D_fake.size()
        # deblur = F.interpolate(deblur,size=(imageH,imageW))
        return -self.D_fake.mean()
        # return -(self.D_fake.mean()+0.05*(self.D_fake*deblur).mean())

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty

    def get_loss(self, net, fakeB, realB,defocus_map):
        defocus_map=defocus_map.detach()
        deblur, _ = defocus_map.abs().max(dim=1, keepdim=True)
        
        self.D_fake = net.forward(fakeB.detach())
        _,_,imageH,imageW = self.D_fake.size()
        deblur = F.interpolate(deblur,size=(imageH,imageW))
        
        # self.D_fake = self.D_fake.mean()

        # Real
        self.D_real = net.forward(realB)
        # self.D_real = self.D_real.mean()
        # Combined loss
        self.loss_D_o = self.D_fake - self.D_real
        self.loss_D = 0.1*(self.loss_D_o*deblur).mean() +self.loss_D_o.mean()
        gradient_penalty = self.calc_gradient_penalty(net, realB.data, fakeB.data)
        return self.loss_D + gradient_penalty

class RelativisticDiscLossLS(nn.Module):
    def name(self):
        return 'RelativisticDiscLossLS'

    def __init__(self):
        super(RelativisticDiscLossLS, self).__init__()

        self.criterionGAN = GANLoss(use_l1=True)
        self.fake_pool = ImagePool(50)  # create image buffer to store previously generated images
        self.real_pool = ImagePool(50)

    def get_g_loss(self, net, fakeB, realB):
        # First, G(A) should fake the discriminator
        self.pred_fake = net.forward(fakeB)

        # Real
        self.pred_real = net.forward(realB)
        errG = (torch.mean((self.pred_real - torch.mean(self.fake_pool.query()) + 1) ** 2) +
                torch.mean((self.pred_fake - torch.mean(self.real_pool.query()) - 1) ** 2)) / 2
        return errG

    def get_loss(self, net, fakeB, realB):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        self.fake_B = fakeB.detach()
        self.real_B = realB
        self.pred_fake = net.forward(fakeB.detach())
        self.fake_pool.add(self.pred_fake)

        # Real
        self.pred_real = net.forward(realB)
        self.real_pool.add(self.pred_real)

        # Combined loss
        self.loss_D = (torch.mean((self.pred_real - torch.mean(self.fake_pool.query()) - 1) ** 2) +
                       torch.mean((self.pred_fake - torch.mean(self.real_pool.query()) + 1) ** 2)) / 2
        return self.loss_D

    def __call__(self, net, fakeB, realB):
        return self.get_loss(net, fakeB, realB)

#用一下doubleGAN
def get_fullD (n_layer=5):
    model_d = NLayerDiscriminator(n_layers=n_layer,
                                  use_sigmoid=False)
    return model_d

class DicsriminatorTail(nn.Module):
    def __init__(self, nf_mult, n_layers, ndf=64, norm_layer=nn.BatchNorm2d, use_parallel=True):
        super(DicsriminatorTail, self).__init__()
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence = [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, norm_layer=nn.InstanceNorm2d, use_parallel=False):
        super(MultiScaleDiscriminator, self).__init__()
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        self.scale_one = nn.Sequential(*sequence)
        self.first_tail = DicsriminatorTail(nf_mult=nf_mult, n_layers=3)
        nf_mult_prev = 4
        nf_mult = 8

        self.scale_two = nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True))
        nf_mult_prev = nf_mult
        self.second_tail = DicsriminatorTail(nf_mult=nf_mult, n_layers=4)
        self.scale_three = nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True))
        self.third_tail = DicsriminatorTail(nf_mult=nf_mult, n_layers=5)

    def forward(self, input):
        x = self.scale_one(input)
        x_1 = self.first_tail(x)
        x = self.scale_two(x)
        x_2 = self.second_tail(x)
        x = self.scale_three(x)
        x = self.third_tail(x)
        return [x_1, x_2, x]

def get_discriminator(discriminator_name):
    #discriminator_name = model_config['d_name']
    if discriminator_name == 'no_gan':
        model_d = None
    elif discriminator_name == 'patch_gan':
        model_d = NLayerDiscriminator(input_nc=3,
                                      n_layers=3,                                     
                                      use_sigmoid=False)
        #model_d = nn.DataParallel(model_d)
    elif discriminator_name == 'double_gan':
        patch_gan = NLayerDiscriminator(input_nc=3,
                                      n_layers=3,                                     
                                      use_sigmoid=False)
        patch_gan = patch_gan.cuda()
        full_gan = get_fullD(n_layer=5)
        full_gan = full_gan.cuda()
        model_d = {'patch': patch_gan,
                   'full': full_gan}
    elif discriminator_name == 'multi_scale':
        model_d = MultiScaleDiscriminator()
        #model_d = nn.DataParallel(model_d)
    else:
        raise ValueError("Discriminator Network [%s] not recognized." % discriminator_name)

    return model_d

class GANTrainer(object):
    def __init__(self, net_d, criterion):
        self.net_d = net_d
        self.criterion = criterion

    def loss_d(self, pred, gt):
        pass

    def loss_g(self, pred, gt):
        pass

    def get_params(self):
        pass

class DoubleGAN(GANTrainer):
    def __init__(self, net_d, criterion):
        GANTrainer.__init__(self, net_d, criterion)
        self.patch_d = net_d['patch'].cuda()
        self.full_d = net_d['full'].cuda()
        self.full_criterion = copy.deepcopy(criterion)

    def loss_d(self, pred, gt):
        return (self.criterion(self.patch_d, pred, gt) + self.full_criterion(self.full_d, pred, gt)) / 2

    def loss_g(self, pred, gt):
        return (self.criterion.get_g_loss(self.patch_d, pred, gt) + self.full_criterion.get_g_loss(self.full_d, pred,
                                                                                                  gt)) / 2

    def get_params(self):
        return list(self.patch_d.parameters()) + list(self.full_d.parameters())

    class Factory:
        @staticmethod
        def create(net_d, criterion): return DoubleGAN(net_d, criterion)