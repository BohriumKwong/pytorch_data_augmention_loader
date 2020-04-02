# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 06:31:10 2020

@author: Bohrium.Kwong
"""

import torch.utils.data as data
import torchvision.transforms.functional as functional
from torchvision.datasets.folder import has_file_allowed_extension

import sys,os
import random
import PIL.Image as Image

class augmention_dataset(data.Dataset):
    """An augmention data loader base on generic data loader \
    where the samples are arranged in this way: ::

        sub_dir/class_x/TFS{'.jpg','.png','.bmp','.tif',...}
        sub_dir/class_x/GHM{'.jpg','.png','.bmp','.tif',...}
        sub_dir/class_x/RTY{'.jpg','.png','.bmp','.tif',...}

        sub_dir/class_y/QDD{'.jpg','.png','.bmp','.tif',...}
        sub_dir/class_y/FGK{'.jpg','.png','.bmp','.tif',...}
        sub_dir/class_y/DGF{'.jpg','.png','.bmp','.tif',...}

    You also can pass a list as one of args which include all the image whole path like this: ::
        ['sub_dir/class_x/GGE.tif', 
         'sub_dir/class_x/GGK.tif', 
         'sub_dir/class_y/KNT.tif',
         'sub_dir/class_y/DNV.tif',
         #...#
         ]
        
    Args:
        sub_dir (string): Root directory path.
        class_to_idx (dict): Dict with items (class_name, class_index).When the Args is None,it will \
            be created automatic base on sub_dir(Args sub_dir and class_to_idx can not both be None)
        image_list: List that include all the image whole path.When the Args is None,it will \
            be created automatic base on sub_dir(Args sub_dir and class_to_idx can not both be None).\
            P.S: the path of image in list can be different from each other (even if sub_dir)
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        repeat_flag: the values of how many times of your data expansion
            by functional.adjust_hue augmention 

     Attributes:
        classes (list): List of the class names.
        samples (list): List of image sample path 
        targets (int): The class_index value for each image 
    """
    def __init__(self, sub_dir = None,class_to_idx = None, image_list = None, transform=None):
        if class_to_idx is None:
            if sub_dir is not None:
                if sys.version_info >= (3, 5):
                # Faster and available in Python 3.5 and above
                    classes = [d.name for d in os.scandir(sub_dir) if d.is_dir()]
                else:
                    classes = [d for d in os.listdir(sub_dir) if os.path.isdir(os.path.join(sub_dir, d))]
                classes.sort()
                class_to_idx = {classes[i]: i for i in range(len(classes))}
            else:
                raise ValueError("Args of sub_dir and class_to_idx can not both be None!")
                
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']        
        if image_list is None:
            if sub_dir is not None:
                image_list = []
                for class_name in sorted(os.listdir(sub_dir)):
                    d = os.path.join(sub_dir, class_name)
                    for root, _, fnames in sorted(os.walk(d)):
                        s = 0
                        for fname in sorted(fnames):
                            if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                                image_list.append(os.path.join(root, fname))
                                s += 1
                        print('Finded class:{},total {} files.'.format(class_name,s))
            else:
                raise ValueError("Args of sub_dir and image_list can not both be None!")
        
        print('Number of images: {}'.format(len(image_list)))
        
        self.class_to_idx = class_to_idx
        self.image_list = image_list
        self.samples = image_list
        self.transform = transform
        self.mode = 1
        
    def setmode(self,mode=1):
        if mode in [1,2]:
            self.mode = mode
        """mode: while training it need to be set mode =2, then will return all images and target after\
                data expansion with augmention method; while just predicting, it need to be set mode =1, \
                in data loader's iterator. More details in the `__getitem__` method
        """
    def shuffle_data(self,shuffle_flag = False):
        """shuffle_flag:bool,shuffle input data or not,it is better to shuffle data begin training
           P.S: You can shuffle data in each training epochs 
        """
        if shuffle_flag:
            if self.mode == 1:
                self.image_list = random.sample(self.image_list, len(self.image_list))
            elif self.mode == 2:
                self.samples = random.sample(self.samples, len(self.samples))
        
    def maketraindata(self, repeat=0):
        """repeat: the values of how many times of your data expansion
        by `functional.adjust_hue augmention`. 0 stands for no expansion.
        """
    #repeat这个参数用于是否对采样进行复制,如果进行复制,\
    #   就会在下面的_getitem_方法中对重复的样本进行不一样的颜色增强
        if self.mode == 2:
            if abs(repeat) == 0:
                self.samples = [(image_path,0) for image_path in self.image_list]
            else:
                repeat = abs(repeat) if repeat % 2 == 1 else abs(repeat) + 1
                self.samples = [(image_path,0) for image_path in self.image_list]
                for y in range(-100,int(100 + repeat/2),int(100*2/repeat)):
                    self.samples = self.samples + \
                        [(image_path,y/1000) for image_path in self.image_list]
                
                self.shuffle_data(True)
    
    def __getitem__(self,index):
        if self.mode == 1:
        # mode =1 为预测时使用,会返回全部的图像和相应的label(如果有label的话,没有就返回-1代替)
            img = Image.open(self.image_list[index])
            if img.size != (224,224):
                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            if self.image_list[index].split("/")[-2] in self.class_to_idx.keys():
                return img,self.class_to_idx[self.image_list[index].split("/")[-2]]
            else:
                return img,-1
                # -1 stands for lack of original label

        
        elif self.mode == 2:
        # mode =2 为训练时使用,会返回增强后全部的图像和相应的label
            image_path,h_value = self.samples[index]
            img = Image.open(image_path)
            if img.size != (224,224):
                img = img.resize((224,224),Image.BILINEAR)           
            if image_path.split("/")[-3].find('COLORNORM') == -1:
                # 该条件针对有颜色转换的数据集使用,这里默认颜色转换的数据集文件夹名字带'COLORNORM'
                #   针对这部分已经执行颜色转换的图片就不再进行颜色增强
                if h_value > 0:
                    hue_factor = random.uniform(h_value,0.1) 
                    img = functional.adjust_hue(img,hue_factor)
                elif h_value < 0:                
                    hue_factor = random.uniform(-0.1,h_value)    
                    img = functional.adjust_hue(img,hue_factor)
            if self.transform is not None:
                img = self.transform(img)
                                            
            return img,self.class_to_idx[image_path.split("/")[-2]]
    
    def __len__(self):
        if self.mode == 2:
            return len(self.samples)
        else:
            return len(self.image_list)
            
if __name__ == '__main__':
#    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    import torch
    import torchvision.transforms as transforms
    import numpy as np
    from tqdm import tqdm as tqdm
    
    train_trans = transforms.Compose([transforms.RandomVerticalFlip(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()
                                        ])
    sub_dir = 'dataloader_test'
    train_dset = augmention_dataset(sub_dir = sub_dir,class_to_idx = None, image_list = None,
                                    transform=train_trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=32, shuffle=False,
        num_workers=0, pin_memory=False)
    
    train_dset.shuffle_data(True)
    train_dset.setmode(2)
    train_dset.maketraindata(3)
    #以下是测试dataloader的demo
    with tqdm(train_loader, desc = 'Augmention data_loader testing', \
            file=sys.stdout) as iterator:
        for i, (input, target) in enumerate(iterator):
            input = input.cpu().numpy()
            target = np.mean(target.cpu().numpy())
            info = (input.shape,target)
            iterator.set_postfix_str('test info :' + str(info))
        iterator.set_postfix_str('Bacth count is ' + str(i+1))

