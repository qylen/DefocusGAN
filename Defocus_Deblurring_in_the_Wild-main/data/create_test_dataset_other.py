from torch.utils.data import Dataset
import os
import torch
import torchvision.transforms.functional as TF
import cv2
import torch.nn.functional as F


class TestDataset(Dataset):
    def __init__(self, testopt):
        super(TestDataset, self).__init__()
        path = testopt['dataroot']

        combine_imgs = os.path.join(os.path.expanduser(path), testopt["combine_name"])
        combine_source_imgs = os.path.join(os.path.expanduser(path), testopt["combine_source_name"])

        combine_imgs = [os.path.join(combine_imgs, os_dir) for os_dir in os.listdir(combine_imgs)]
        combine_source_imgs = [os.path.join(combine_source_imgs, os_dir) for os_dir in os.listdir(combine_source_imgs)]
        combine_imgs.sort()
        combine_source_imgs.sort()

        self.uegt_imgs = {}
        for c_img,cs_img in zip(combine_imgs,combine_source_imgs):
            self.uegt_imgs[c_img] = [cs_img]

        self.uegt_imgs = [(key, values) for key, values in self.uegt_imgs.items()]
        self.uegt_imgs = sorted(self.uegt_imgs, key=lambda x: x[0])

    def __len__(self):
        return len(self.uegt_imgs)

    def __getitem__(self, index):
        c_img, (cs_img) = self.uegt_imgs[index]
        c_img_name = c_img
        gt_img = cv2.imread(c_img, -1)
        cs_img = cv2.imread(cs_img[0], -1)
        
        gt_img = torch.tensor(gt_img / 65535.0).float().permute(2, 0, 1)
        cs_img = torch.tensor(cs_img / 65535.0).float().permute(2, 0, 1)

        if cs_img.size(1) > cs_img.size(2):
            
            gt_img = TF.rotate(gt_img, 90)
            cs_img = TF.rotate(cs_img, 90)

        return gt_img,cs_img, os.path.basename(c_img_name).split('.')

class TestDatasetPixelDP(Dataset):
    def __init__(self, testopt):
        super(TestDatasetPixelDP, self).__init__()
        path = testopt['dataroot']

        combine_source_imgs = os.path.join(os.path.expanduser(path), testopt["combine_source_name"])

        combine_source_imgs = [os.path.join(combine_source_imgs, os_dir) for os_dir in os.listdir(combine_source_imgs)]
        combine_source_imgs.sort()

        self.uegt_imgs = {}
        for c_img,cs_img in zip(combine_source_imgs,combine_source_imgs):
            self.uegt_imgs[c_img] = [cs_img]

        self.uegt_imgs = [(key, values) for key, values in self.uegt_imgs.items()]
        self.uegt_imgs = sorted(self.uegt_imgs, key=lambda x: x[0])

    def __len__(self):
        return len(self.uegt_imgs)

    def __getitem__(self, index):
        c_img, (cs_img) = self.uegt_imgs[index]
        c_img_name = c_img
        cs_img = cv2.imread(c_img, -1)
        
        
        cs_img = torch.tensor(cs_img / 65535.0).float().permute(2, 0, 1)

        cs_flag = 0
        if cs_img.size(1) > cs_img.size(2):  
            cs_flag = 1         
            cs_img = cs_img.permute(0,2,1)
        a=cs_img.size(1)
        b=cs_img.size(2)
        if cs_img.size(1)!= 1120:
            cs_img=F.pad(cs_img,pad=[0,0,0,1120-cs_img.size(1)])

        return cs_img, os.path.basename(c_img_name).split('.'),a,b,cs_flag

class TestDatasetCUHK(Dataset):
    def __init__(self, testopt):
        super(TestDatasetCUHK, self).__init__()
        path = testopt['dataroot']

        # combine_imgs = os.path.join(os.path.expanduser(path), testopt["combine_name"])
        combine_source_imgs = os.path.join(os.path.expanduser(path), testopt["combine_source_name"])

        #combine_imgs = [os.path.join(path, os_dir) for os_dir in os.listdir(path)]
        combine_source_imgs = [os.path.join(combine_source_imgs, os_dir) for os_dir in os.listdir(combine_source_imgs)]
        combine_source_imgs.sort()

        self.uegt_imgs = {}
        for c_img,cs_img in zip(combine_source_imgs,combine_source_imgs):
            self.uegt_imgs[c_img] = [cs_img]

        self.uegt_imgs = [(key, values) for key, values in self.uegt_imgs.items()]
        self.uegt_imgs = sorted(self.uegt_imgs, key=lambda x: x[0])

    def __len__(self):
        return len(self.uegt_imgs)

    def __getitem__(self, index):
        c_img, (cs_img) = self.uegt_imgs[index]
        c_img_name = c_img
        cs_img = cv2.imread(c_img, -1)       
        cs_img = torch.tensor(cs_img / 255.0).float().permute(2, 0, 1)
        cs_flag = 0

        if cs_img.size(1) > cs_img.size(2):  
            cs_flag = 1         
            cs_img = cs_img.permute(0,2,1)
        a=cs_img.size(1)
        b=cs_img.size(2)
        
        if cs_img.size(1)<480:
            cs_img=F.pad(cs_img,pad=[0,0,0,480-cs_img.size(1)])
        if cs_img.size(1)>480 and cs_img.size(1)<640:
            cs_img=F.pad(cs_img,pad=[0,0,0,640-cs_img.size(1)])
        if cs_img.size(2)<480:
            cs_img=F.pad(cs_img,pad=[0,480-cs_img.size(2),0,0])
        if cs_img.size(2)>480 and cs_img.size(2)<640:
            cs_img=F.pad(cs_img,pad=[0,640-cs_img.size(2),0,0])


        return cs_img, os.path.basename(c_img_name).split('.'),a,b,cs_flag

class TestDatasetDeblur(Dataset):
    def __init__(self, testopt):
        super(TestDatasetDeblur, self).__init__()
        path = testopt['datarootDC']

        recover_imgs = os.path.join(os.path.expanduser(path), testopt["recover_name"])
        combine_imgs = os.path.join(os.path.expanduser(path), testopt["AoF_name"])

        recover_imgs = [os.path.join(recover_imgs, os_dir) for os_dir in os.listdir(recover_imgs)]
        combine_imgs = [os.path.join(combine_imgs, os_dir) for os_dir in os.listdir(combine_imgs)]
        
        recover_imgs.sort()
        combine_imgs.sort()

        self.uegt_imgs = {}
        for recover_img, c_img in zip(recover_imgs, combine_imgs):
            self.uegt_imgs[c_img] = [recover_img]

        self.uegt_imgs = [(key, values) for key, values in self.uegt_imgs.items()]
        self.uegt_imgs = sorted(self.uegt_imgs, key=lambda x: x[0])

    def __len__(self):
        return len(self.uegt_imgs)

    def __getitem__(self, index):
        c_img, (recover_img) = self.uegt_imgs[index]
        c_img_name = c_img
        recover_img_name = recover_img[0]
        recover_img = cv2.imread(recover_img[0], -1)
        
        gt_img = cv2.imread(c_img, -1)
        
        recover_img = torch.tensor(recover_img / 255.0).float().permute(2, 0, 1)
        
        gt_img = torch.tensor(gt_img / 65535.0).float().permute(2, 0, 1)
        

        if recover_img.size(1) > recover_img.size(2):
            recover_img = TF.rotate(recover_img, 90)
            
            gt_img = TF.rotate(gt_img, 90)
            

        return recover_img, gt_img, os.path.basename(c_img_name).split('.')

class TestDataset2(Dataset):
    def __init__(self, testopt):
        super(TestDataset, self).__init__()
        path = testopt['dataroot']

        left_imgs = os.path.join(os.path.expanduser(path), testopt["left_name"])
        right_imgs = os.path.join(os.path.expanduser(path), testopt["right_name"])
        combine_imgs = os.path.join(os.path.expanduser(path), testopt["combine_name"])
        combine_source_imgs = os.path.join(os.path.expanduser(path), testopt["combine_source_name"])
        

        left_imgs = [os.path.join(left_imgs, os_dir) for os_dir in os.listdir(left_imgs)]
        right_imgs = [os.path.join(right_imgs, os_dir) for os_dir in os.listdir(right_imgs)]
        combine_imgs = [os.path.join(combine_imgs, os_dir) for os_dir in os.listdir(combine_imgs)]
        combine_source_imgs = [os.path.join(combine_source_imgs, os_dir) for os_dir in os.listdir(combine_source_imgs)]
        left_imgs.sort()
        right_imgs.sort()
        combine_imgs.sort()
        combine_source_imgs.sort()

        self.uegt_imgs = {}
        for l_img, r_img, c_img,cs_img in zip(left_imgs, right_imgs, combine_imgs,combine_source_imgs):
            self.uegt_imgs[c_img] = [l_img, r_img,cs_img]

        self.uegt_imgs = [(key, values) for key, values in self.uegt_imgs.items()]
        self.uegt_imgs = sorted(self.uegt_imgs, key=lambda x: x[0])

    def __len__(self):
        return len(self.uegt_imgs)

    def __getitem__(self, index):
        c_img, (l_img, r_img,cs_img) = self.uegt_imgs[index]
        c_img_name = c_img
        l_img = cv2.imread(l_img, -1)
        r_img = cv2.imread(r_img, -1)
        gt_img = cv2.imread(c_img, -1)
        cs_img = cv2.imread(cs_img, -1)
        l_img = torch.tensor(l_img / 65535.0).float().permute(2, 0, 1)
        r_img = torch.tensor(r_img / 65535.0).float().permute(2, 0, 1)
        gt_img = torch.tensor(gt_img / 65535.0).float().permute(2, 0, 1)
        cs_img = torch.tensor(cs_img / 65535.0).float().permute(2, 0, 1)

        if l_img.size(1) > l_img.size(2):
            l_img = TF.rotate(l_img, 90)
            r_img = TF.rotate(r_img, 90)
            gt_img = TF.rotate(gt_img, 90)
            cs_img = TF.rotate(cs_img, 90)

        return l_img, r_img, gt_img,cs_img, os.path.basename(c_img_name).split('.')