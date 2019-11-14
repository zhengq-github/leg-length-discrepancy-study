# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 16:33:09 2018

@author: ZhengQ
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb
import time
from dataloader import *
import copy
import numpy as np
from matplotlib.pyplot import contour, contourf
import matplotlib.pyplot as plt


class FCNmodel_3pool(nn.Module):
    def __init__(self,n_class):
        super(FCNmodel_3pool,self).__init__()
        self.n_class3 = n_class
        
        self.conv = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.ResBlock1 = ResBlock(32,64).cuda()
        self.ResBlock2 = ResBlock(64,128).cuda()
        self.ResBlock3 = ResBlock(128,256).cuda()
        self.ResBlock4 = ResBlock(256,512).cuda()
        
        self.deconv1   = nn.ConvTranspose2d(512,256,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.ResBlock_r1 = ResBlock(256,256).cuda()
        self.deconv_classifier1   = nn.ConvTranspose2d(256,n_class,kernel_size=9,stride=8,padding=1,dilation=1,output_padding=1)

        self.deconv2   = nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.ResBlock_r2 = ResBlock(128,128).cuda()
        self.deconv_classifier2   = nn.ConvTranspose2d(128,n_class,kernel_size=5,stride=4,padding=1,dilation=1,output_padding=1)

        self.deconv3   = nn.ConvTranspose2d(128, 64,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.ResBlock_r3 = ResBlock(64,64).cuda()
        self.deconv_classifier3 = nn.ConvTranspose2d(64,n_class,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        
        self.deconv4   = nn.ConvTranspose2d(64, 32,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.ResBlock_r4 = ResBlock(32,32).cuda()
        self.deconv_classifier4 = nn.Conv2d(32,n_class,kernel_size=1)
        
        

    def forward(self,x):

        
        
                                    # x:  batch*1*448*160
        x1 = self.conv(x)           # x1: batch*32*448*160
        x2 = self.pool(x1)          # x2: batch*32*224*80
           
        x3 = self.ResBlock1(x2)     # x3: batch*64*224*80
        x4 = self.pool(x3)          # x4: batch*64*112*40
        
        x5 = self.ResBlock2(x4)     # x5: batch*128*112*40
        x6 = self.pool(x5)          # x6: batch*128*56*20
        
        x7 = self.ResBlock3(x6)     # x7: batch*256*56*20
        x8 = self.pool(x7)          # x8: batch*256*28*10
        
        x9 = self.ResBlock4(x8)     # x9: batch*512*28*10
        
        #deconv
        y1 = self.deconv1(x9)        # y1: batch*256*56*20
        y1s = y1 + x7                # y1s: batch*256*56*20
        y2 = self.ResBlock_r1(y1s)     # y2: batch*256*56*20
            
        y3 = self.deconv2(y2)        # y3: batch*128*112*40
        y3s = y3 + x5                # y3s: batch*128*112*40
        y4 = self.ResBlock_r2(y3s)     # y4: batch*128*112*40
        
        y5 = self.deconv3(y4)        # y5: batch*64*224*80
        y5s = y5 + x3                # y5s: batch*64*224*80
        y6 = self.ResBlock_r3(y5s)     # y6: batch*64*224*80
        
        y7 = self.deconv4(y6)        # y7: batch*32*448*160
        y7s = y7 + x1                # y7s: batch*32*448*160
        y8 = self.ResBlock_r4(y7s)     # y8: batch*32*448*160
        
       
        z1 = self.deconv_classifier1(y2)
        z2 = self.deconv_classifier2(y4)
        z3 = self.deconv_classifier3(y6)    
        z4 = self.deconv_classifier4(y8)     

        z = z1 + z2 + z3 + z4
      
        
        return z


class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ResBlock,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.convShortConnect = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
        
    def forward(self,x):
                                 # x: batch*channels*rx*ry*rz
        x1 = self.conv1(x)    
        x2 = self.bn1(x1)
        x3 = self.relu1(x2)
        
        x4 = self.conv2(x3)
        x5 = self.bn2(x4)

        x5s = self.convShortConnect(x)
        x6 = x5s + x5
        
        x7 = self.relu2(x6)
        
        return x7
    



if __name__ == '__main__':
    __spec__ = None
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()   # + delete intermediate variable
    print('remember release GPU memory')

    plt.ion()
    num_workers      = 1
    train_batch_size = 1
    test_batch_size  = 1
    rx,ry            = 448,160 #48,48,48  #80,80,120
    lr               = 1e-2
    w_decay          = 1e-5    
    sliceNum         = 16     #24
    n_class          = 2
    
    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))

    print('use_gpu:',use_gpu)
    print('num_gpu:',num_gpu)

    
    
    fcn_model = FCNmodel_3pool(n_class)
    fcn_model.cuda()
    criterion = nn.BCEWithLogitsLoss()
    criterion.cuda()
    optimizer = optim.SGD(fcn_model.parameters(),lr=lr,momentum=0.9)
    
    train_data = dataloader(training=True)
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=train_batch_size,shuffle=False,num_workers=num_workers)
    
    test_data = dataloader(training=False)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=test_batch_size,shuffle=False,num_workers=num_workers)

    fcn_model.train()
    for i,batch in enumerate(test_loader):

        if i == 0:
            input0  = batch['img'].cuda()
            y0      = batch['target'].cuda()

            for iter in range(1):
                optimizer.zero_grad()
                output4 = fcn_model(input0)
                loss = criterion(output4,y0)   
                
                loss.backward()
                print("iter={},      loss={}".format(iter,loss.cpu().item()))
                optimizer.step()
                
            with torch.no_grad():
                fcn_model.eval()
                output = fcn_model(input0).cpu().numpy()
                pred = output.transpose(0,2,3,1).reshape(-1,n_class).argmax(axis=1).reshape(rx,ry)
                lab = 255*pred
                lab_pre = lab
                
                img = input0.cpu().numpy()
                inp = 255*img[0,0,...]
                
#                lab_expert = y0_n4[0,...].cpu().numpy()
#                lab1 = np.zeros_like(lab_expert)
#                lab1[np.where(lab_expert==1)]=1
#                lab2 = np.zeros_like(lab_expert)
#                lab2[np.where(lab_expert==2)]=1
#                lab3 = np.zeros_like(lab_expert)
#                lab3[np.where(lab_expert==3)]=1
#                lab4 = np.zeros_like(lab_expert)
#                lab4[np.where(lab_expert==4)]=1
                
                
                plt.figure()
                plt.imshow(inp,cmap='gray')
#                contour(lab1,colors='red')
#                contour(lab2,colors='blue')
#                contour(lab3,colors='yellow')
#                contour(lab4,colors='green')
                contour(lab_pre,colors='white')
                plt.title('expert')
                plt.axis('off')

                plt.ioff()
                plt.show()
            break

    torch.cuda.empty_cache()
    print('remember release GPU memory')




                # lab = 255*pred[10,:,:]
                # pred = output.numpy()[0,1,...]
                # pred = 255*(pred-pred.min())/(pred.max()-pred.min())




            # s0 = y[:,0,...].sum()
            # s1 = y[:,1,...].sum()
            # s2 = s0 + s1
            # weights = torch.FloatTensor([s0/s2,s1/s2])
            # weight = weights[y[:,1,...].long()]

            # weight = torch.stack([weights[y[:,1,...].long()],weights[y[:,1,...].long()]],dim=1)

            # weight = weights[y[:,1,...].long()]

            # pdb.set_trace()









            # s0 = y0[:,0,...].sum()
            # s1 = y0[:,1,...].sum()
            # s2 = s0 + s1
            # weights0 = torch.FloatTensor([s0/s2,s1/s2])  # weight for 0 > weight for 1
            # weights1 = torch.FloatTensor([s1/s2,s0/s2])  # weight for 0 < weight for 1

            # weight = torch.stack([weights0[y[:,0,...].long()],weights1[y[:,1,...].long()]],dim=1)
















#    fcn_model = FCNmodel_3pool(n_class)
#    fcn_model.cuda()
#    input0 = torch.randn(1,1,rx,ry).type(torch.FloatTensor).cuda()
#    y0 = torch.empty(1,rx,ry,dtype=torch.long).random_(n_class).cuda()
#    criterion = nn.CrossEntropyLoss()
#    criterion.cuda()
#    optimizer = optim.SGD(fcn_model.parameters(),lr=lr,momentum=0.9)

#    fcn_model.train()
#    for iter in range(10):
#        optimizer.zero_grad()
#        output0 = fcn_model(input0)
#        loss = criterion(output0,y0)   
#        loss.backward()
#        print("iter={},      loss={}".format(iter,loss.cpu().item()))
#        optimizer.step()
        
#    torch.cuda.empty_cache()
#    print('remember release GPU memory')
















        
        ##################################################################################
#        
#        a0 = torch.cat((x,y7),dim=1)
#        a1 = self.conv2(a0)           # x1: batch*64*448*160
#        a2 = self.pool(a1)          # x2: batch*64*224*80
#           
#        a3 = self.ResBlock1(a2)     # x3: batch*128*224*80
#        a4 = self.pool(a3)          # x4: batch*128*112*40
#        
#        a5 = self.ResBlock2(a4)     # x5: batch*256*112*40
#        a6 = self.pool(a5)          # x6: batch*256*56*20
#        
#        a7 = self.ResBlock3(a6)     # x7: batch*512*56*20
#        
#        #deconv
#        b1 = self.deconv1(a7)        # y1: batch*256*112*40
#        b1s = b1 + a5                # y1s: batch*256*112*40
#        b2 = self.ResBlock4(b1s)     # y2: batch*256*112*40
#        
#        b3 = self.deconv2(b2)        # y3: batch*128*224*80
#        b3s = b3 + a3                # y3s: batch*128*224*80
#        b4 = self.ResBlock5(b3s)     # y4: batch*128*224*80
#        
#        b5 = self.deconv3(b4)        # y5: batch*64*448*160
#        b5s = b5 + a1                # y5s: batch*64*448*160
#        b6 = self.ResBlock6(b5s)     # y6: batch*64*448*160
#        
#        b7_1 = self.deconv_classifier1_n2(b2)
#        b7_2 = self.deconv_classifier2_n2(b4)
#        b7_3 = self.deconv_classifier3_n2(b6)     # y7: batch*5*448*160     
#        
#        b7 = b7_1 + b7_2 + b7_3
#        
#        
#        ##################################################################################
#        c0 = torch.cat((x,y7),dim=1)
#        c1 = self.conv3(c0)           # x1: batch*64*448*160
#        c2 = self.pool(c1)          # x2: batch*64*224*80
#           
#        c3 = self.ResBlock1(c2)     # x3: batch*128*224*80
#        c4 = self.pool(c3)          # x4: batch*128*112*40
#        
#        c5 = self.ResBlock2(c4)     # x5: batch*256*112*40
#        c6 = self.pool(c5)          # x6: batch*256*56*20
#        
#        c7 = self.ResBlock3(c6)     # x7: batch*512*56*20
#        
#        #deconv
#        d1 = self.deconv1(c7)        # y1: batch*256*112*40
#        d1s = d1 + c5                # y1s: batch*256*112*40
#        d2 = self.ResBlock4(d1s)     # y2: batch*256*112*40
#        
#        d3 = self.deconv2(d2)        # y3: batch*128*224*80
#        d3s = d3 + c3                # y3s: batch*128*224*80
#        d4 = self.ResBlock5(d3s)     # y4: batch*128*224*80
#        
#        d5 = self.deconv3(d4)        # y5: batch*64*448*160
#        d5s = d5 + c1                # y5s: batch*64*448*160
#        d6 = self.ResBlock6(d5s)     # y6: batch*64*448*160
#        
#        d7_1 = self.deconv_classifier1_n3(d2)
#        d7_2 = self.deconv_classifier2_n3(d4)
#        d7_3 = self.deconv_classifier3_n3(d6)     # y7: batch*5*448*160     
#        
#        d7 = d7_1 + d7_2 + d7_3
#        
#        ##################################################################################
#        e0 = torch.cat((x,y7,b7,d7),dim=1)
#        e1 = self.conv4(e0)           # x1: batch*64*448*160
#        e2 = self.pool(e1)          # x2: batch*64*224*80
#           
#        e3 = self.ResBlock1(e2)     # x3: batch*128*224*80
#        e4 = self.pool(e3)          # x4: batch*128*112*40
#        
#        e5 = self.ResBlock2(e4)     # x5: batch*256*112*40
#        e6 = self.pool(e5)          # x6: batch*256*56*20
#        
#        e7 = self.ResBlock3(e6)     # x7: batch*512*56*20
#        
#        #deconv
#        f1 = self.deconv1(e7)        # y1: batch*256*112*40
#        f1s = f1 + e5                # y1s: batch*256*112*40
#        f2 = self.ResBlock4(f1s)     # y2: batch*256*112*40
#        
#        f3 = self.deconv2(f2)        # y3: batch*128*224*80
#        f3s = f3 + e3                # y3s: batch*128*224*80
#        f4 = self.ResBlock5(f3s)     # y4: batch*128*224*80
#        
#        f5 = self.deconv3(f4)        # y5: batch*64*448*160
#        f5s = f5 + e1                # y5s: batch*64*448*160
#        f6 = self.ResBlock6(f5s)     # y6: batch*64*448*160
#        
#        f7_1 = self.deconv_classifier1_n4(f2)
#        f7_2 = self.deconv_classifier2_n4(f4)
#        f7_3 = self.deconv_classifier3_n4(f6)     # y7: batch*5*448*160     
#        
#        f7 = f7_1 + f7_2 + f7_3
        
        
        ##################################################################################







































