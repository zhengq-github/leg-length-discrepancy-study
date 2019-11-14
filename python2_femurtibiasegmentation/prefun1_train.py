
print('~~~~~~~~~~~~~~prefun~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~prefun~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~prefun~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~prefun~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~prefun~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~prefun~~~~~~~~~~~~~')

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
#from torch.nn import functional as F
import numpy as np
import torch
import torchvision
from torchvision import datasets,models,transforms
#import matplotlib.pyplot as plt
import copy
from dataloader import *
from FCNmodel_3pool import *
#from matplotlib.pyplot import contour, contourf
#import time
import pdb
import os
import shutil
#from PIL import Image
import nibabel as nib
import scipy.misc as misc
import imageio
from scipy.misc import imresize


n_class         = 3
batch_size_train  = 10
batch_size_test   = 10
epochs          = range(0,151)
lr           = 1e-4
momentum     = 0.9
L2_factor    = 1e-5
L1_factor    = 1e-5
step_size    = 1000
gamma        = 0.1

root = 'E:/project_chop/project_leglength/code_for_training/dataTrain/data5_deep_learning/segmentation/'
model_dir = root + "models_3" #segmentation"
score_dir = root + "scores_3" #segmentation"   #os.path.join(model_dir,configs)
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)   #os.remove delete file       rmtree delete folder
if os.path.exists(score_dir):
    shutil.rmtree(score_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

train_data = dataloader(training=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size =batch_size_train , shuffle = False, num_workers = 0)
len_traindata = len(train_loader)
test_data = dataloader(training=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size =batch_size_test , shuffle = False, num_workers = 0)
len_testdata   = len(test_loader)


#dir_model = model_dir + "\\model_epoch400"
#fcn_model = torch.load(dir_model)

fcn_model = FCNmodel_3pool(n_class)
fcn_model.cuda()
criterion = nn.CrossEntropyLoss()
criterion.cuda()
optimizer = optim.SGD(fcn_model.parameters(),lr=lr,momentum=momentum,weight_decay=L2_factor)
scheduler = lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)


use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))
print('use_gpu:',use_gpu)
print('num_gpu:',num_gpu)
#pdb.set_trace()

def train_model():
    fcn_model.train()
    for epoch in epochs:
        scheduler.step()
        
        train_data = dataloader(training=True)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size =batch_size_train , shuffle = True, num_workers = 0)
        len_traindata = len(train_loader)
        

        loss_running = 0
        for iter,batch in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = batch['img'].cuda()
            labels = batch['mask'].cuda()

            outputs = fcn_model(inputs)

            loss = criterion(outputs,labels)   
            loss.backward()
            
            optimizer.step()
            loss_running += loss.data.cpu().item() * batch_size_train
            print("epoch={},iter={},loss={}".format(epoch,iter,loss.data.cpu().item()))
            del loss
            
        loss_epoch = loss_running / len_traindata
        print("~~train data~~~~~~~loss_epoch"+str(epoch)+"={}".format(loss_epoch))
        np.save(score_dir + "\\loss_epoch" + str(epoch),loss_epoch)
        del loss_epoch
        del loss_running

        if epoch >= 10 and epoch % 10 == 0:
            torch.save(fcn_model,model_dir + "\\model_epoch" + str(epoch))
            with torch.no_grad():
                val_model(epoch)
            
def iou(pred,target):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        ious.append(float(intersection) / union)
    return ious


def val_model(epoch):
    fcn_model.eval()
    total_ious = []
    for iter,batch in enumerate(test_loader):
            inputs,labels,width0 = batch['img'].cuda(),batch['mask'],batch['width']
            outputs = fcn_model(inputs).data.cpu().numpy()
            width1 = width0.numpy()

            N,_,h,w = outputs.shape
            pred = outputs.transpose(0,2,3,1).reshape(-1,n_class).argmax(axis=1).reshape(N,h,w)#.astype(np.int32)

            target = labels.numpy()#.reshape(N,h,w)
            
            for p,t in zip(pred,target):
                total_ious.append(iou(p,t))
            
            mask_path = batch['mask_path']
            if epoch > 0:
                for i in range(N):
                    
                    width2 = width1[i,0]
                    pred0 = 100*imresize(pred[i,...].astype(np.uint8),(448,width2),interp='nearest')
                    
                    mask_path0 = mask_path[i]
                    mask_path1 = mask_path0[:-4] + "_pred.png"
                    misc.imsave(mask_path1,pred0)
                    
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    np.save(score_dir + "\\dice_epoch" + str(epoch),ious)
    print("~~~~~~~~~~~~~~~~test data~~~~ious_epoch"+str(epoch)+"={}".format(ious))



if __name__ == "__main__":
    __spec__ = None
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()   # release cache +++++++++ delete intermediate variable

    with torch.no_grad():
        val_model(0) # show the accuarcy before training

    train_model()
    

    
 



