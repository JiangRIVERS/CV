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

        self.dowsample=downsample
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
        residual=x
        output=self.conv1(x)
        output=self.bn1(output)
        output=self.relu(output)

        output=self.conv2(output)
        output=self.bn2(output)
        output=self.relu(output)

        output=self.conv3(output)
        output=self.bn3(output)
        if self.dowsample:
            residual=self.dowsample(x)

        output+=residual
        output=self.relu(output)

        return output

class FPN(nn.Module):

    def __init__(self,block,blocks):
        super(FPN,self).__init__()

        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        #Bottom-up layers
        self.layer1=self._make_layer(block,64,64,blocks[0],stride=1)
        self.layer2=self._make_layer(block,256,128,blocks[1],stride=2)
        self.layer3 = self._make_layer(block, 512, 256, blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, 512, blocks[3], stride=2)

        #Top layer
        self.toplayer=nn.Conv2d(2048,256,kernel_size=1,stride=1,padding=0) #To reduce channels

        #Laterak layers
        self.lateral4=nn.Conv2d(1024,256,kernel_size=1,stride=1,padding=0)
        self.lateral3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.lateral2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        #Smooth layers
        self.smooth5=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)


    def _make_layer(self,block,inplanes,planes,blocks,stride):
        layers=[]
        layers.append(block(inplanes,planes,stride,downsample=True))
        for i in range(1,blocks):
            layers.append(block(block.expansion*planes,planes,stride=1))

        return nn.Sequential(*layers)

    def forward(self,x):
        #Bottom-up
        c1=self.conv1(x)
        c1=self.bn1(c1)
        c1=self.relu(c1)
        c1=self.maxpool(c1)

        c2=self.layer1(c1)
        c3=self.layer2(c2)
        c4=self.layer3(c3)
        c5=self.layer4(c4)
        #Top-bottom
        m5=self.toplayer(c5)
        m4=self.lateral4(c4)+F.interpolate(m5,self.lateral4(c4).size()[2:],mode="bilinear",align_corners=True)
        m3 = self.lateral3(c3) + F.interpolate(m4, self.lateral3(c3).size()[2:], mode="bilinear", align_corners=True)
        m2 = self.lateral2(c2) + F.interpolate(m3, self.lateral2(c2).size()[2:], mode="bilinear", align_corners=True)
        #Smooth
        p5=self.smooth5(m5)
        p6=self.maxpool(p5)
        p4 = self.smooth4(m4)
        p3 = self.smooth3(m3)
        p2 = self.smooth2(m2)

        return p2,p3,p4,p5

def ResNet101_FPN():
    return FPN(Bottleneck,[3,4,23,3])

if __name__=="__main__":
    m=ResNet101_FPN()
    a=torch.rand(1,3,600,900)
    output=m(a)
    for i in output:
        print(i.size())






