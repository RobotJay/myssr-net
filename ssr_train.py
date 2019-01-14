#coding=utf-8
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.ssrdataset import collate_fn, dataset
import torch
from model.SSRNET64 import MySSRNet
from torchvision.models import DenseNet
import torch.utils.data as torchdata
from utils.train import train,trainlog
from torchvision import datasets, models, transforms
from torchvision.models import resnet50,resnet101,resnet152,resnet18,vgg16
import torch.optim as optim
from torch.optim import lr_scheduler

from  torch.nn import CrossEntropyLoss,MSELoss
import logging
from utils.data_aug import *
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for mae and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=90,
                        help="number of epochs")
    parser.add_argument("--netType1", type=int, required=True,
                        help="network type 1")
    parser.add_argument("--netType2", type=int, required=True,
                        help="network type 2")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="validation split ratio")

    args = parser.parse_args()
    return args

def main():
    # args = get_args()
    # netType1 = args.netType1
    # netType2 = args.netType2


    netType1 = 1
    netType2 = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    save_dir = '/home/heils-server/User/gwb/pytorch_classification-master/log/ssr/age/mae_ter'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = '%s/trainlog.log'%save_dir
    trainlog(logfile)

    rawdata_root = '/home/heils-server/User/gwb/pytorch_classification-master/input/'
    all_pd = pd.read_table("/home/heils-server/User/gwb/pytorch_classification-master/input/data.csv",sep=",",
                         header=None, names=['ImageName', 'label'])

    train_pd, val_pd = train_test_split(all_pd, test_size=0.2, random_state=43)
                                        # stratify=all_pd['label'])
    print(val_pd.shape)

    '''数据扩增'''
    data_transforms = {
        'train': Compose([
            # RandomRotate(angles=(-15,15)),
            # RandomResizedCrop(size=(256, 256)),
            # ExpandBorder(size=(256, 256), resize=True),
            Resize((64,64)),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': Compose([
            # ExpandBorder(size=(256, 256), resize=True),
            Resize((64,64)),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_set = {}
    data_set['train'] = dataset(imgroot=rawdata_root,anno_pd=train_pd,
                               transforms=data_transforms["train"],
                               )
    data_set['val'] = dataset(imgroot=rawdata_root,anno_pd=val_pd,
                               transforms=data_transforms["val"],
                               )


    dataloader = {}
    dataloader['train']=torch.utils.data.DataLoader(data_set['train'], batch_size=200,
                                                   shuffle=True, num_workers=4,collate_fn=collate_fn)
    dataloader['val']=torch.utils.data.DataLoader(data_set['val'], batch_size=200,
                                                   shuffle=True, num_workers=4,collate_fn=collate_fn)
    '''model'''

    stage_num = [3, 3, 3]
    lambda_local = 0.25 * (netType1 % 5)
    lambda_d = 0.25 * (netType2 % 5)

    model = MySSRNet(stage_num, lambda_local, lambda_d)

    base_lr = 0.0001

    resume =None
    if resume:
        logging.info('resuming finetune from %s'%resume)
        model.load_state_dict(torch.load(resume))
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-5)  
    from loss.criteria import MAELoss
    criterion = MAELoss()

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)




    train(model,
          epoch_num=500,
          start_epoch=0,
          optimizer=optimizer,
          criterion=criterion,
          exp_lr_scheduler=exp_lr_scheduler,
          data_set=data_set,
          data_loader=dataloader,
          save_dir=save_dir,
          print_inter=50,
          val_inter=500)

if __name__ == '__main__':
    main()
