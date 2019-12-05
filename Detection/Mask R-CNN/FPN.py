"""
Filename: FPN.py

A Pytorch implementation of ResNet101_FPN

@author: Jiang Rivers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion=4

    def __init__(self,inplanes,planes,stride=1,downsample=False):
        super(Bottleneck, self).__init__()
        self.conv1=nn.Conv2d(inplanes,planes,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)
        self.conv3=nn.Conv2d(planes,self.expansion*planes,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn3=nn.BatchNorm2d(self.expansion*planes)
        self.relu = nn.ReLU(inplace=True)

        if downsample:
            self.dowsample=nn.Sequential(
                nn.Conv2d(inplanes,self.expansion*planes,kernel_size=1,stride=stride,padding=0,bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self,x):




