from torch.utils.data import Dataset
import os
import json
import torch
import torchvision.transforms.functional as TF
import cv2
import numpy as np
import random


def read_frame(frame, rotate = None):
    norm_val= 2**16-1
    if rotate is not None:
        frame = cv2.rotate(frame, rotate)
    frame = frame / norm_val
    frame = frame[...,::-1]

    return np.expand_dims(frame, axis = 0)

def color_to_gray(img):
    c_linear = 0.2126*img[:, :, 0] + 0.7152*img[:, :, 1] + 0.07228*img[:, :, 2]
    c_linear_temp = c_linear.copy()

    c_linear_temp[np.where(c_linear <= 0.0031308)] = 12.92 * c_linear[np.where(c_linear <= 0.0031308)]
    c_linear_temp[np.where(c_linear > 0.0031308)] = 1.055 * np.power(c_linear[np.where(c_linear > 0.0031308)], 1.0/2.4) - 0.055

    img[:, :, 0] = c_linear_temp
    img[:, :, 1] = c_linear_temp
    img[:, :, 2] = c_linear_temp

    return img

def crop_multi(x, hrg, wrg, is_random=False, row_index=0, col_index=1):

    h, w = x[0].shape[row_index], x[0].shape[col_index]

    if (h <= hrg) or (w <= wrg):
        raise AssertionError("The size of cropping should smaller than the original image")

    if is_random:
        h_offset = int(np.random.uniform(0, h - hrg) - 1)
        w_offset = int(np.random.uniform(0, w - wrg) - 1)
        results = []
        for data in x:
            results.append(data[int(h_offset):int(hrg + h_offset), int(w_offset):int(wrg + w_offset)])
        return np.asarray(results)
    else:
        # central crop
        h_offset = (h - hrg) / 2
        w_offset = (w - wrg) / 2
        results = []
        for data in x:
            results.append(data[int(h_offset):int(h - h_offset), int(w_offset):int(w - w_offset)])
        return np.asarray(results)

class TrainDataset(Dataset):
    def __init__(self, trainopt):
        super(TrainDataset, self).__init__()
        path = trainopt['dataroot']
        self.image_size = trainopt['image_size']
        self.batch_size = trainopt['iter_size']
        self.max_iter = trainopt["max_iter"]
        self.epoch_num = 0
        trainpairs_forreading = str(trainopt['trainpairs'])
        self.h = 256
        self.w = 256
        if not os.path.exists(trainpairs_forreading):
            left_imgs = os.path.join(os.path.expanduser(path), trainopt["left_name"])
            right_imgs = os.path.join(os.path.expanduser(path), trainopt["right_name"])
            #blur_imgs = os.path.expanduser(trainopt["blur_name"])
            blur_imgs = os.path.join(os.path.expanduser(path), trainopt["blur_name"])
            combine_imgs = os.path.join(os.path.expanduser(path), trainopt["combine_name"])
            combine_source_imgs = os.path.join(os.path.expanduser(path), trainopt["combine_source_name"])

            left_imgs = [os.path.join(left_imgs, os_dir) for os_dir in os.listdir(left_imgs)]
            right_imgs = [os.path.join(right_imgs, os_dir) for os_dir in os.listdir(right_imgs)]
            blur_imgs = [os.path.join(blur_imgs, os_dir) for os_dir in os.listdir(blur_imgs)]
            combine_imgs = [os.path.join(combine_imgs, os_dir) for os_dir in os.listdir(combine_imgs)]
            combine_source_imgs = [os.path.join(combine_source_imgs, os_dir) for os_dir in os.listdir(combine_source_imgs)]
            
            left_imgs.sort()
            right_imgs.sort()
            blur_imgs.sort()
            combine_imgs.sort()
            combine_source_imgs.sort()

            self.uegt_imgs = {}
            for l_img, r_img, c_img, b_img,cs_img in zip(left_imgs, right_imgs, combine_imgs, blur_imgs,combine_source_imgs):
                self.uegt_imgs[c_img] = [l_img, r_img, b_img,cs_img]
            with open(trainpairs_forreading, 'w') as f:
                json.dump(self.uegt_imgs, f)
        else:
            with open(trainpairs_forreading, 'r') as f:
                self.uegt_imgs = json.load(f)
        self.uegt_imgs = [(key, values) for key, values in self.uegt_imgs.items()]
        self.uegt_imgs = sorted(self.uegt_imgs, key=lambda x: x[0])

    def __len__(self):
        return len(self.uegt_imgs)

    def random_augmentation(self, under):
        c, w, h = under.shape
        w_start = w - self.image_size
        h_start = h - self.image_size

        random_w = 1 if w_start <= 1 else torch.randint(low=1, high=w_start, size=(1, 1)).item()
        random_h = 1 if h_start <= 1 else torch.randint(low=1, high=h_start, size=(1, 1)).item()
        return random_w, random_h
    
    # def __getitem__(self, index): #使用数据增强
    #     self.is_augment = False
    #     c_img, (l_img, r_img, b_img,cs_img) = self.uegt_imgs[index]
    #     l_img = cv2.imread(l_img, -1)
    #     r_img = cv2.imread(r_img, -1)
    #     b_img = cv2.imread(b_img, -1)
    #     gt_img = cv2.imread(c_img, -1)
    #     cs_img = cv2.imread(cs_img, -1)

    #     l_frame = read_frame(l_img)
    #     r_frame = read_frame(r_img)
    #     b_frame = read_frame(b_img)
    #     gt_frame = read_frame(gt_img)
    #     c_frame = read_frame(cs_img)

        

    #     if self.is_augment:
    #         # Noise
    #         if random.uniform(0, 1) <= 0.05:
    #         # if random.uniform(0, 1) >= 0.0:
    #             row,col,ch = l_frame[0].shape
    #             mean = 0.0
    #             sigma = random.uniform(0.001, (0.005**(1/2)))
    #             gauss = np.random.normal(mean,sigma,(row,col,ch))
    #             gauss = gauss.reshape(row,col,ch)

    #             l_frame = np.expand_dims(np.clip(l_frame[0] + gauss, 0.0, 1.0), axis = 0)
    #             r_frame = np.expand_dims(np.clip(r_frame[0] + gauss, 0.0, 1.0), axis = 0)
    #             c_frame = np.expand_dims(np.clip(c_frame[0] + gauss, 0.0, 1.0), axis = 0)

    #         # Grayscale
    #         if random.uniform(0, 1) <= 0.3:
    #             l_frame = np.expand_dims(color_to_gray(l_frame[0]), axis = 0)
    #             r_frame = np.expand_dims(color_to_gray(r_frame[0]), axis = 0)
    #             c_frame = np.expand_dims(color_to_gray(c_frame[0]), axis = 0)
    #             gt_frame = np.expand_dims(color_to_gray(gt_frame[0]), axis = 0)
    #             b_frame = np.expand_dims(color_to_gray(b_frame[0]), axis = 0)


    #         # Scaling
    #         if random.uniform(0, 1) <= 0.5:
    #             scale = random.uniform(0.7, 1.0)
    #             row,col,ch = l_frame[0].shape

    #             l_frame = np.expand_dims(cv2.resize(l_frame[0], dsize=(int(col*scale), int(row*scale)), interpolation=cv2.INTER_AREA), axis = 0)
    #             r_frame = np.expand_dims(cv2.resize(r_frame[0], dsize=(int(col*scale), int(row*scale)), interpolation=cv2.INTER_AREA), axis = 0)
    #             c_frame = np.expand_dims(cv2.resize(c_frame[0], dsize=(int(col*scale), int(row*scale)), interpolation=cv2.INTER_AREA), axis = 0)
    #             gt_frame = np.expand_dims(cv2.resize(gt_frame[0], dsize=(int(col*scale), int(row*scale)), interpolation=cv2.INTER_AREA), axis = 0)
    #             b_frame = np.expand_dims(cv2.resize(b_frame[0], dsize=(int(col*scale), int(row*scale)), interpolation=cv2.INTER_AREA), axis = 0)
    #     cropped_frames = np.concatenate([l_frame, r_frame, c_frame, gt_frame,b_frame], axis = 3)
        
    #     cropped_frames = crop_multi(cropped_frames, self.h, self.w, is_random = True)
        
    #     l_patches = cropped_frames[:, :, :, :3]
    #     shape = l_patches.shape
    #     h = shape[1]
    #     w = shape[2]
    #     l_patches = l_patches.reshape((h, w, -1, 3))
    #     l_img = torch.FloatTensor(np.transpose(l_patches, (2, 3, 0, 1)))

    #     r_patches = cropped_frames[:, :, :, 3:6]
    #     r_patches = r_patches.reshape((h, w, -1, 3))
    #     r_img = torch.FloatTensor(np.transpose(r_patches, (2, 3, 0, 1)))

    #     c_patches = cropped_frames[:, :, :, 6:9]
    #     c_patches = c_patches.reshape((h, w, -1, 3))
    #     cs_img = torch.FloatTensor(np.transpose(c_patches, (2, 3, 0, 1)))

    #     gt_patches = cropped_frames[:, :, :, 9:12]
    #     gt_patches = gt_patches.reshape((h, w, -1, 3))
    #     gt_img = torch.FloatTensor(np.transpose(gt_patches, (2, 3, 0, 1)))

    #     b_patches = cropped_frames[:, :, :, 12:15]
    #     b_patches = b_patches.reshape((h, w, -1, 3))
    #     b_img = torch.FloatTensor(np.transpose(b_patches, (2, 3, 0, 1)))


    #     ## todo
    #     # data augmentation

    #     b_img = b_img * 25
    #     # print(b_img.max(),b_img.min(),b_img.mean())
    #     # _, s_w, s_h = b_img.shape

    #     return torch.squeeze(l_img),torch.squeeze(r_img), torch.squeeze(b_img), torch.squeeze(gt_img),torch.squeeze(cs_img)

    def __getitem__(self, index): #没有用数据增强
        c_img, (l_img, r_img, b_img,cs_img) = self.uegt_imgs[index]
        l_img = cv2.imread(l_img, -1)
        r_img = cv2.imread(r_img, -1)
        b_img = cv2.imread(b_img, -1)
        gt_img = cv2.imread(c_img, -1)
        cs_img = cv2.imread(cs_img, -1)

        l_img = torch.tensor(l_img/65535.).float().permute(2, 0, 1)
        r_img = torch.tensor(r_img / 65535.).float().permute(2, 0, 1)
        b_img = torch.tensor(b_img / 65535.).float().permute(2, 0, 1)
        gt_img = torch.tensor(gt_img / 65535.).float().permute(2, 0, 1)
        cs_img = torch.tensor(cs_img / 65535.).float().permute(2, 0, 1)

        ## todo
        # data augmentation

        if l_img.size(2) < l_img.size(1):
            l_img = TF.rotate(l_img, 90)
            r_img = TF.rotate(r_img, 90)
            gt_img = TF.rotate(gt_img, 90)
            cs_img = TF.rotate(cs_img, 90)

        b_img = b_img * 25
        # print(b_img.max(),b_img.min(),b_img.mean())
        _, s_w, s_h = b_img.shape

        return l_img, r_img, b_img, gt_img,cs_img



class ValDataset(Dataset):
    def __init__(self, valopt):
        super(ValDataset, self).__init__()
        path = valopt['dataroot']
        left_imgs = os.path.join(os.path.expanduser(path), valopt["left_name"])
        right_imgs = os.path.join(os.path.expanduser(path), valopt["right_name"])
        blur_imgs = os.path.join(os.path.expanduser(path), valopt["blur_name"])
        combine_imgs = os.path.join(os.path.expanduser(path), valopt["combine_name"])
        combine_source_imgs = os.path.join(os.path.expanduser(path), valopt["combine_source_name"])

        left_imgs = [os.path.join(left_imgs, os_dir) for os_dir in os.listdir(left_imgs)]
        right_imgs = [os.path.join(right_imgs, os_dir) for os_dir in os.listdir(right_imgs)]
        blur_imgs = [os.path.join(blur_imgs, os_dir) for os_dir in os.listdir(blur_imgs)]
        combine_imgs = [os.path.join(combine_imgs, os_dir) for os_dir in os.listdir(combine_imgs)]
        combine_source_imgs = [os.path.join(combine_source_imgs, os_dir) for os_dir in os.listdir(combine_source_imgs)]

        left_imgs.sort()
        right_imgs.sort()
        blur_imgs.sort()
        combine_imgs.sort()
        combine_source_imgs.sort()

        self.uegt_imgs = {}
        for l_img, r_img, c_img, b_img,cs_img in zip(left_imgs, right_imgs, combine_imgs, blur_imgs,combine_source_imgs):
            self.uegt_imgs[c_img] = [l_img, r_img, b_img,cs_img]

        self.uegt_imgs = [(key, values) for key, values in self.uegt_imgs.items()]
        self.uegt_imgs = sorted(self.uegt_imgs, key=lambda x: x[0])

    def __len__(self):
        return len(self.uegt_imgs)

    def __getitem__(self, index):
        c_img, (l_img, r_img, b_img,cs_img) = self.uegt_imgs[index]
        c_img_name = c_img

        l_img = cv2.imread(l_img, -1)
        r_img = cv2.imread(r_img, -1)
        b_img = cv2.imread(b_img, -1)
        gt_img = cv2.imread(c_img, -1)
        cs_img = cv2.imread(cs_img, -1)

        l_img = torch.tensor(l_img / 65535.).float().permute(2, 0, 1)
        r_img = torch.tensor(r_img / 65535.).float().permute(2, 0, 1)
        b_img = torch.tensor(b_img / 65535.).float().permute(2, 0, 1)
        gt_img = torch.tensor(gt_img / 65535.).float().permute(2, 0, 1)
        cs_img = torch.tensor(cs_img / 65535.).float().permute(2, 0, 1)

        if l_img.size(2) < l_img.size(1):
            l_img = TF.rotate(l_img, 90)
            r_img = TF.rotate(r_img, 90)
            gt_img = TF.rotate(gt_img, 90)
            cs_img = TF.rotate(cs_img, 90)

        b_img = b_img * 25

        return l_img, r_img, b_img, gt_img,cs_img, os.path.basename(c_img_name).split('.')

