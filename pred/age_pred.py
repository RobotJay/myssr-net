import logging
import torch
import cv2
from torch.autograd import Variable
import numpy as np
from math import ceil
import pandas as pd
import os
import torch.utils.data as torchdata
from sklearn.model_selection import train_test_split
# from model.MySSRNET import MySSRNet
# from model.MySSRNET_gen import MySSRNet_gen
from model.SSRNET64 import MySSRNet,MySSRNet_gen
# from model.MySSRNET92 import MySSRNet,MySSRNet_gen
from utils.data_aug import *
from utils.bd_tiangong_dataset import collate_fn, dataset
stage_num = [3,3,3]
lambda_local = 0.25
lambda_d = 0.25

model_age = MySSRNet(stage_num, lambda_local, lambda_d)
# model_gen = MySSRNet_gen(stage_num,lambda_d,lambda_d)
#
resume_age = '/home/gwb/pycharm/project/myssr_net/log/asina/08/260---[6.6351].pth'
if resume_age:
    logging.info('resuming finetune from %s' % resume_age)
    model_age.load_state_dict(torch.load(resume_age,map_location=lambda  storage,loc:storage))


data = pd.read_csv('/home/gwb/pycharm/project/myssr_net/input/web/camera.csv',sep=",",
                         header=None, names=['ImageName', 'label','pred'])


# from model.mobile import MobileNet
# model_age = MobileNet(input_size = (64,64))
# resume_age = '/home/gwb/pycharm/project/SSR-Net-master/demo/log/asina/06/1480---[4.0193].pth'
# model_age.load_state_dict(torch.load(resume_age, map_location=lambda storage, loc: storage))

rawdata_root = '/home/gwb/pycharm/project/myssr_net/input/web/camera/'

'''数据扩增'''
data_transforms = {
    'test': Compose([
        #RandomRotate(angles=(-20,20)),
        # RandomResizedCrop(size=(64, 64)),
        #Resize((64,64)),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
}

data_set = {}
data_set['test'] = dataset(imgroot=rawdata_root,anno_pd=data,
                           transforms=data_transforms["test"],
                           )

data_loader = {}
data_loader['test'] = torchdata.DataLoader(data_set['test'], batch_size=1, num_workers=1,
                                           shuffle=False, pin_memory=False)

model_age = model_age
model_age.eval()

from loss.criteria import MAELoss
criterion = MAELoss()

if not os.path.exists('./web/camera'):
    os.makedirs('./web/camera')

test_size = ceil(len(data_set['test']) / data_loader['test'].batch_size)
test_preds = np.zeros((len(data_set['test'])), dtype=np.float32)
true_label = np.zeros((len(data_set['test'])), dtype=np.int)
idx = 0
loss = []
for batch_cnt_test, data_test in enumerate(data_loader['test']):
    inputs, labels = data_test
    inputs = Variable(inputs)
    labels = Variable(torch.from_numpy(np.array(labels)).float())
    # forward
    outputs = model_age(inputs)
    print("outputs;\t", outputs)

    # testloss = criterion(outputs, labels)

    # loss.append(testloss.detach().numpy().tolist())
    # print(testloss)
    # testloss += testloss.data[0]
    test_preds[idx:(idx + labels.size(0))] = outputs.cpu().detach().numpy()
    true_label[idx:(idx + labels.size(0))] = labels.data.cpu().detach().numpy()
    # statistics
    idx += labels.size(0)

# ['ImageName', 'label','pred']
test_pred = data[['ImageName']].copy()
test_pred['label'] = list(true_label)
test_pred['pred'] = list(test_preds)
#test_pred['label'] = test_pred['label'].apply(lambda x: label2class[int(x)])
test_pred[['ImageName',"label",'pred']].to_csv('web/camera/{0}.csv'.format('age'),sep=",",index=False)

mean_loss = []
for i in loss:
    mean_loss.append(i)

print(sum(mean_loss)/len(mean_loss))
