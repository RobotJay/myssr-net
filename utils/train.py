#coding=utf8
from __future__ import division
import torch
import os,time,datetime
from torch.autograd import Variable
import logging
import numpy as np
from math import ceil

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def dt():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def trainlog(logfilepath, head='%(message)s'):
    logger = logging.getLogger('mylogger')
    logging.basicConfig(filename=logfilepath, level=logging.INFO, format=head)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def train(model,epoch_num,start_epoch,optimizer,criterion,exp_lr_scheduler,
          data_set,data_loader,save_dir,print_inter=200,val_inter=200):
    # Train the model
    step = -1
    total_step = len(data_loader['train'])
    for epoch in range(epoch_num):
        for i, (images, labels) in enumerate(data_loader['train']):
            step += 1
            images = images.to(device)
            labels = torch.from_numpy(np.array(labels)).float()
            labels = labels.to(device)
            
            outputs = model(images).to(device)
           
            trainloss = criterion(outputs, labels)

            
            optimizer.zero_grad()
            trainloss.backward()
            optimizer.step()

            if (i + 1) % 1000 == 0:
                save_path = os.path.join(save_dir,'%d-[%.4f].pth' % (epoch,trainloss))
                
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epoch_num, i + 1, total_step, trainloss.item()))

            if step % val_inter == 0:
                # Test the model
                model.eval() 
                with torch.no_grad():
                    for images, labels in data_loader['val']:
                        images = images.to(device)
                        labels = torch.from_numpy(np.array(labels)).float()
                        labels = labels.to(device)
                        outputs = model(images)
                        testloss = criterion(outputs, labels)
                        
    
            save_path = os.path.join(save_dir,
                                     '%d-[%.4f].pth' % (epoch,testloss))
            torch.save(model.state_dict(), save_path)
            




