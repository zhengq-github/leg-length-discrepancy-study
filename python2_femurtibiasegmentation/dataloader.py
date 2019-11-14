# -*- coding: utf-8 -*-
"""
Created on Fri May 18 10:15:29 2018

@author: ZhengQ
"""

import random
import torch
import numpy as np
import torch.utils.data as data
import glob
from PIL import Image
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from matplotlib.pyplot import contour, contourf
#from skimage import measure    #mask0 = measure.label(mask,4)
#import scipy.io as sio
#from sklearn.model_selection import train_test_split
#import copy
import pdb
from scipy.misc import imresize
from skimage.transform import resize

root = 'E:/project_chop/project_leglength/code_for_training/dataTrain/data5_deep_learning/segmentation/'

class dataloader(data.Dataset):
    def __init__(self,training=True):
        super().__init__()
        
        self.training=training
        
        img1_train = glob.glob(root + 'train_From1To90/*_img1.png')
        img2_train = glob.glob(root + 'train_From1To90/*_img2.png')
        img3_train = glob.glob(root + 'train_From1To90/*_img3.png')
        img4_train = glob.glob(root + 'train_From1To90/*_img4.png')
        img5_train = glob.glob(root + 'train_From1To90/*_img5.png')
        img6_train = glob.glob(root + 'train_From1To90/*_img6.png')
        
        img1_test = glob.glob(root + 'test_From91To131/*_img1.png')
        img2_test = glob.glob(root + 'test_From91To131/*_img2_flip.png')

        
        mask1_train = glob.glob(root + 'train_From1To90/*_lab1.png')
        mask2_train = glob.glob(root + 'train_From1To90/*_lab2.png')
        mask3_train = glob.glob(root + 'train_From1To90/*_lab3.png')
        mask4_train = glob.glob(root + 'train_From1To90/*_lab4.png')
        mask5_train = glob.glob(root + 'train_From1To90/*_lab5.png')
        mask6_train = glob.glob(root + 'train_From1To90/*_lab6.png')
        
        mask1_test  = glob.glob(root + 'test_From91To131/*_lab1.png')
        mask2_test  = glob.glob(root + 'test_From91To131/*_lab2_flip.png')

        img_train = img1_train + img2_train + img3_train + img4_train +img5_train + img6_train
        img_test  = img1_test  + img2_test
        
        mask_train = mask1_train + mask2_train + mask3_train + mask4_train + mask5_train + mask6_train
        mask_test  = mask1_test  + mask2_test


        if training:
            self.img_path = img_train
            self.mask_path = mask_train

        else:
            self.img_path = img_test
            self.mask_path = mask_test
        
    def __getitem__(self,index):
        
        img_path  = self.img_path[index]
        mask_path = self.mask_path[index]
        img0  = np.array(Image.open(img_path))
        mask0 = np.array(Image.open(mask_path))
        
        height,width0 = mask0.shape
        input_h = 224 #448
        input_w = 112 #80
        
        
        mask1 = np.zeros_like(mask0)
        mask1[np.where(mask0==100)] = 1
        mask1[np.where(mask0==200)] = 2

        img0 = imresize(img0,(input_h,input_w),interp='bilinear')
        mask2 = imresize(mask1,(input_h,input_w),interp='nearest')


        target0 = np.zeros((3,input_h,input_w))
        for c in range(3):
            target0[c][mask2==c] = 1
        
        mask = torch.from_numpy(mask2).long()
        target = torch.from_numpy(target0).type(torch.FloatTensor)
        
        img1  = torch.from_numpy(img0/255)
        img = img1.unsqueeze(0).type(torch.FloatTensor)


        width = torch.from_numpy(np.array([width0]))
            
        sample = {'img':img,'img_path':img_path,'mask_path':mask_path,'target':target,'mask':mask,'width':width}
        
        return sample
    
    def __len__(self):
        return len(self.img_path)
    


if __name__ == '__main__':
    
#    plt.ion()   # interactive mode
#    train_data = dataloader_2(training=True)
#    train_loader = torch.utils.data.DataLoader(train_data, batch_size =1 , shuffle = True, num_workers = 0)
    
    test_data = dataloader(training=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size =1 , shuffle = False, num_workers = 0)
    
        
    for i,batch in enumerate(test_loader):
        print(i,batch['img'].size(),batch['target'].size())
        
        if i == 0:
            img  = 255*batch['img'].numpy()[0,0,...]
            target = batch['mask'].numpy()
            img_path = batch['img_path'][0]
            print(img_path)
            plt.figure()
            plt.imshow(img, cmap='gray')
            contour(target[0,...],colors='red')
            plt.title('Batch from dataloader')
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
            
        
        
        
#    for i in range(len(test_data)):
#        sample = test_data[i]
#        print(i,sample['img'].size(),sample['target'].size())
#            lab0 = measure.label(mask,4)
#            lab1 = np.zeros_like(lab0)
#            lab1[np.where(lab0==1)] = 1
#            lab2 = np.zeros_like(lab0)
#            lab2[np.where(lab0==2)] = 2
#            lab3 = np.zeros_like(lab0)
#            lab3[np.where(lab0==3)] = 3
#            lab4 = np.zeros_like(lab0)
#            lab4[np.where(lab0==4)] = 4
            
#            control_path = batch['control_path']
#            xCon = batch['xCon'].numpy()
#            xCase = batch['xCase'].numpy()
#            name0 = batch['name']
#            name1 = copy.deepcopy(name0)
#            name = str(name1)[3:-2]
#            
#            lenCon = batch['lenCon'].numpy()
#            img_path = batch['img_path']
#            label_path = batch['label_path']
#            
#            break
            
#            img_batch = batch['img']
#            inp = img_batch[0,:,:,:]
#            inp = inp.numpy().transpose((1,2,0))
#            mean=np.array([0.485,0.456,0.406])
#            std=np.array([0.229,0.224,0.225])
#            inp = std*inp + mean
#            inp = np.clip(inp,0,1)
#            
#            label_batch = batch['target']
#            lab = label_batch[0,1,:,:]
#            lab = lab.numpy()
#            
#            plt.figure()
#            plt.imshow(inp)
#            contour(lab)
#            plt.title('Batch from dataloader')
#            plt.axis('off')
#            plt.ioff()
#            plt.show()
#            break
    
            
            
            
            
            
            
            
###################################################################    
#    print(len(train_loader),len(test_loader))
    
#    aa = next(iter(train_loader))
#    img = aa['img'].numpy()[0,:,:,:].transpose((1,2,0))
#    mean=np.array([0.485,0.456,0.406])
#    std=np.array([0.229,0.224,0.225])
#    img = std*img + mean
#    img = np.clip(img,0,1)
#    label = aa['target'].numpy()[0,1,:,:]
#    
#    plt.ion()   # interactive mode
#    plt.figure()
#    p1 = plt.subplot(121)
#    p2 = plt.subplot(122)
#    p1.imshow(img)
#    p2.imshow(label)
#    
#    plt.ioff()
#    plt.show()













