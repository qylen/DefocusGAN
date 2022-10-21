import argparse
import os
import numpy
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from utils import util, build_code_arch
from data.create_coc_datasetours import TrainDataset
from models.blur_kernel_model import KernelEDNet
from loss.blur_loss_alternate import GemoLoss,GemorestructNet
import time

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, default='/home/zjc/dual_pixel/Defocus_Deblurring_in_the_Wild-main/option/train/COC_Dataset_Train.yaml', help='Defocus Deblur: Path to option ymal file.')
train_args = parser.parse_args()

opt, resume_state = build_code_arch.build_resume_state(train_args)
opt, logger, tb_logger = build_code_arch.build_logger(opt)

for phase, dataset_opt in opt['dataset'].items():
    if phase == 'train':
        train_dataset = TrainDataset(dataset_opt)
        train_loader = DataLoader(
            train_dataset, batch_size=dataset_opt['batch_size'], shuffle=True,
            num_workers=dataset_opt['workers'], pin_memory=True)
        logger.info('Number of train images: {:,d}'.format(len(train_dataset)))
assert train_loader is not None

# create model
model = KernelEDNet()
model_R= GemorestructNet()
model = model.cuda()
model_R = model_R.cuda()
# for p in model_R.parameters():
#     print(type(p),p.size())
optimizer = Adam(model.parameters(), betas=(opt['train']['beta1'], opt['train']['beta2']),
                 lr=opt['train']['lr'])
optimizer_R= Adam(model_R.parameters(), betas=(opt['train']['beta1'], opt['train']['beta2']),
                 lr=opt['train']['lr'])                 

scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer,
                                     milestones=opt['train']['lr_steps'],
                                     gamma=opt['train']['lr_gamma'])
scheduler_R = lr_scheduler.MultiStepLR(optimizer=optimizer_R,
                                     milestones=opt['train']['lr_steps'],
                                     gamma=opt['train']['lr_gamma'])


# resume training
if resume_state:
    logger.info('Resuming training from epoch: {}.'.format(
        resume_state['epoch']))
    start_epoch = resume_state['epoch']
    optimizer.load_state_dict(resume_state['optimizers'])
    scheduler.load_state_dict(resume_state['schedulers'])
    model.load_state_dict(resume_state['state_dict'])
    if (resume_state['state_dict_Rs'] and resume_state['optimizers_R'] and resume_state['optimizers_R']) is not None:
        model_R.load_state_dict(resume_state['state_dict_Rs'])
        optimizer_R.load_state_dict(resume_state['optimizers_R'])
        scheduler_R.load_state_dict(resume_state['schedulers_R'])
else:
    start_epoch = 0

criterion = GemoLoss()


# model = torch.nn.DataParallel(model)
# training
total_epochs = opt['train']['epoch']

max_steps = len(train_loader)
logger.info('Start training from epoch: {:d}'.format(start_epoch))

current_step = 0
loss_epoch=[]  
for epoch in range(start_epoch, total_epochs + 1):
    criterion.iter = epoch
    starttimeidx=time.time()
    for index, train_data in tqdm(enumerate(train_loader)):
        # training
        l_img, r_img,gt_img,cs_img = train_data
        l_img = l_img.cuda()
        r_img = r_img.cuda()
        gt_img=gt_img.cuda()
        cs_img=cs_img.cuda()
        x = torch.cat((l_img, r_img), dim=1)
        COC_img = model(x)
        re_img = model_R(COC_img[0],gt_img)
        losses = criterion(COC_img[0],re_img,cs_img)
        grad_loss = losses["total_loss"]
        if epoch <10 or (epoch>=15 and epoch<20) or (epoch>=25):
            optimizer.zero_grad()
            grad_loss.backward()
            optimizer.step()
        if (epoch>=10 and epoch<15) or (epoch>=20 and epoch<25):
            optimizer_R.zero_grad()
            grad_loss.backward()
            optimizer_R.step()
        current_step = epoch * max_steps + index
        
        # log
        message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
            epoch, current_step, scheduler.get_last_lr()[0])
        for k, v in losses.items():
            v = v.cpu().item()
            message += '{:s}: {:.4e} '.format(k, v)
            # tensorboard logger
            if opt['use_tb_logger'] and 'debug' not in opt['name']:
                tb_logger.add_scalar(k, v, current_step)
        logger.info(message)
        loss_epoch.append(grad_loss.data.cpu())
    # update learning rate
    scheduler.step()
    
    endtimeidx =time.time()
    if epoch % 1 == 0:
        with util.Logger('./epoch_blur_loss.txt'):
            print('# train # epoch:{:4d} loss{:4f} train_time:{:4f}.'.format(epoch,float(numpy.array(loss_epoch).mean()),float((endtimeidx-starttimeidx)/60)) )
    loss_epoch=[]  
    # save models and training states
    if epoch % opt['logger']['save_checkpoint_freq'] == 0:
        logger.info('Saving models and training states.')
        save_filename = '{}_{}.pth'.format(epoch, 'models')
        save_path = os.path.join(opt['path']['models'], save_filename)
        state_dict = model.state_dict()
        state_dictRs = model_R.state_dict()
        save_checkpoint = {'state_dict': state_dict,
                           'optimizers': optimizer.state_dict(),
                           'schedulers': scheduler.state_dict(),
                           'epoch': epoch,
                           'state_dict_Rs': state_dictRs,
                           'optimizers_R': optimizer_R.state_dict(),
                           'schedulers_R': scheduler_R.state_dict()
                           }
        torch.save(save_checkpoint, save_path)
    torch.cuda.empty_cache()

logger.info('Saving the final model.')
save_filename = 'latest.pth'
save_path = os.path.join(opt['path']['models'], save_filename)
save_checkpoint = {"state_dict": model.state_dict(),
                   'optimizers': optimizer.state_dict(),
                   'schedulers': scheduler.state_dict(),
                   "epoch": opt['train']['epoch']}
torch.save(save_checkpoint, save_path)
logger.info('End of training.')
tb_logger.close()
