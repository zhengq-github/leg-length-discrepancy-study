
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
import timeit

n_class         = 3
batch_size_train  = 10
batch_size_test   = 1
epochs          = range(0,3001)
lr           = 1e-4
momentum     = 0.9
L2_factor    = 1e-5
L1_factor    = 1e-5
step_size    = 2000
gamma        = 0.1

root = 'E:/project_chop/project_leglength/code_for_training/dataTrain/data5_deep_learning/segmentation/'
model_dir = root + "models_3_segmentation"
score_dir = root + "scores_3_segmentation"   #os.path.join(model_dir,configs)

test_data = dataloader(training=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size =batch_size_test , shuffle = False, num_workers = 0)
len_testdata   = len(test_loader)


dir_model = model_dir + "\\model_epoch100"
fcn_model = torch.load(dir_model)

#fcn_model = FCNmodel_3pool(n_class)
cudaNum = torch.device('cuda:1')
fcn_model.cuda(cudaNum)
criterion = nn.CrossEntropyLoss()
criterion.cuda(cudaNum)
optimizer = optim.SGD(fcn_model.parameters(),lr=lr,momentum=momentum,weight_decay=L2_factor)
scheduler = lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)



def iou(pred,target):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        ious.append(float(intersection) / union)
    return ious


if __name__ == "__main__":
    __spec__ = None
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()   # release cache +++++++++ delete intermediate variable
    
    with torch.no_grad():
        fcn_model.eval()
        total_ious = []
        for iter,batch in enumerate(test_loader):
                inputs,labels,width0 = batch['img'].cuda(cudaNum),batch['mask'],batch['width']
#                start1 = timeit.timeit()
                outputs = fcn_model(inputs).data.cpu().numpy()
#                end1 = timeit.timeit()
#                print(end1-start1)
#                pdb.set_trace()
                width1 = width0.numpy()
        
                N,_,h,w = outputs.shape
                pred = outputs.transpose(0,2,3,1).reshape(-1,n_class).argmax(axis=1).reshape(N,h,w)#.astype(np.int32)
        
                target = labels.numpy()#.reshape(N,h,w)
                
                for p,t in zip(pred,target):
                    total_ious.append(iou(p,t))
                
                mask_path = batch['mask_path']
        
                for i in range(N):
                    
                    width2 = width1[i,0]
                    pred0 = 100*imresize(pred[i,...].astype(np.uint8),(448,width2),interp='nearest')
                    
                    mask_path0 = mask_path[i]
                    mask_path1 = mask_path0[:-4] + "_pred.png"
                    misc.imsave(mask_path1,pred0)
                        
    
    
        total_ious = np.array(total_ious).T  # n_class * val_len
        ious = np.nanmean(total_ious, axis=1)
        print("~~~~~~~~test data~~~~ious_epoch==={}".format(ious))












