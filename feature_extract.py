# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:32:42 2020

@author: zhijiezheng
"""

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models
import h5py

TRAIN_ROOT = 'data/train'
TEST_ROOT = 'data/test'
MEAN = [0.485, 0.456, 0.406] # ImageNet数据集的均值
STD = [0.229, 0.224, 0.225] # ImageNet数据集的标准差

def feature_extract(name):
    # 根据模型设置预处理方式
    if name == 'resnet50':
        p = 2048
        MODEL = models.resnet50(pretrained = True)
        print('resnet50 loads finish.')
    
    elif name == 'googlenet':
        p = 1024
        MODEL = models.googlenet(pretrained = True)
        print('googlenet loads finish.')
    
    elif name == 'resnext':
        p = 2048
        MODEL = models.resnext50_32x4d(pretrained = True)
        print('resnext50_32x4d loads finish.')      
    
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                     transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
    
    # 加载数据
    train_data = ImageFolder(root = TRAIN_ROOT, transform = preprocess)
    test_data = ImageFolder(root = TEST_ROOT, transform = preprocess)
    # 对应的label
#    print(train_data.class_to_idx)
    
    # 冻结模型
    for param in MODEL.parameters():
        param.requires_grad = False
        
    # 去掉模型的全连接层
    class Net(torch.nn.Module):
        def __init__(self , model):
            super(Net, self).__init__()
            self.net_layer = torch.nn.Sequential(*list(model.children())[:-1])
    
        def forward(self, x):
            x = self.net_layer(x)
            return x.view(x.shape[0:2])
    
    model = Net(MODEL)
    model.eval()
        
    train_loader = DataLoader(train_data, batch_size = 100, shuffle = False, pin_memory = True, num_workers = 2)
    test_loader = DataLoader(test_data, batch_size = 100, shuffle = False, pin_memory = True, num_workers = 2)
    
    # 初始化要保存的数据
    train = torch.zeros(25000, p)
    test = torch.zeros(12500, p)
    label = torch.zeros(25000,)
    
    # 特征抽取
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(train_loader):
            output = model(batch_x)
            train[i*100:(i+1)*100] = output
            label[i*100:(i+1)*100] = batch_y
        for i, (batch_x, batch_y) in enumerate(test_loader):
            output = model(batch_x)
            test[i*100:(i+1)*100] = output
    
    # 保存模型
    with h5py.File("%s.h5"%name) as h:
        h.create_dataset("train", data = train)
        h.create_dataset("test", data = test)
        h.create_dataset("label", data = label)
        
    
if __name__ == '__main__':
    # 提取特征
    feature_extract('googlenet')
    print('googlenet features extract finish.')
    feature_extract('resnet50')
    print('resnet50 features extract finish.')
    feature_extract('resnext')
    print('resnext features extract finish.')