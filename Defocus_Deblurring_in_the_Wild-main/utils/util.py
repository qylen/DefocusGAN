import os
import logging
from datetime import datetime
import numpy as np
import random
import torch
import math
from kornia.losses import ssim
from skimage import measure
import lpips
import sys
from skimage.metrics import structural_similarity as compare_ssim

from torch.autograd import Variable
from collections import deque

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.squeeze().permute(1, 2, 0).cpu().numpy()
    img2 = img2.squeeze().permute(1, 2, 0).cpu().numpy()
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    # ssim_value = ssim(img1, img2, 11, 'mean')
    # return 1 - ssim_value.item()
    img1 = img1.squeeze().permute(1, 2, 0).cpu().numpy()
    img2 = img2.squeeze().permute(1, 2, 0).cpu().numpy()
    ssim_value = compare_ssim(img1, img2, data_range=1, channel_axis=2)
    return ssim_value




def calculate_mae(img1, img2):
    mae = torch.mean((img1 - img2).abs(), dim=[2, 3, 1])
    return mae.squeeze().item()

def calculate_lpips(img1, img2):
    # img1 = lpips.im2tensor(img1)
    # img2 = lpips.im2tensor(img2)
    img1 = img1 * 2.0 - 1
    img2 = img2 * 2.0 - 1
    spatial = False
    loss_fn_vgg = lpips.LPIPS(net='squeeze',spatial=spatial)
    return loss_fn_vgg.forward(img1, img2).squeeze().item()


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.random.manual_seed(seed)


def squeeze2d(input, factor):
    if factor == 1:
        return input

    B, C, H, W = input.size()

    assert H % factor == 0 and W % factor == 0, "H or W modulo factor is not 0"

    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)

    return x

class Logger():
    def __init__(self, filename):
        self.name = filename
        self.file = open(filename, 'a+', encoding='utf-8')
        self.alive = True
        self.stdout = sys.stdout
        sys.stdout = self

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        if self.alive:
            sys.stdout = self.stdout
            self.file.close()
            self.alive = False

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()




class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.sample_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = deque()

    def add(self, images):
        if self.pool_size == 0:
            return images
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
            else:
                self.images.popleft()
                self.images.append(image)

    def query(self):
        if len(self.images) > self.sample_size:
            return_images = list(random.sample(self.images, self.sample_size))
        else:
            return_images = list(self.images)
        return torch.cat(return_images, 0)