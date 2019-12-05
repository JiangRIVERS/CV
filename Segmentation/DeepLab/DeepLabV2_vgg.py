"""
Filename: DeepLabV2_vgg.py

A DeepLabV2 implementation based on vgg16.

@author Jiang Rivers
"""

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class Astrous(nn.Module):

    def __init__(self,planes,n_classes,rate):
        super(Astrous,self).__init__()
        self.atrous_convolution = nn.Conv2d(512, planes, kernel_size=3,
                                            stride=1, padding=rate, dilation=rate)
        self.fc1=nn.Conv2d(planes,planes,kernel_size=1,stride=1,padding=0)
        self.fc2 = nn.Conv2d(planes, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x=self.atrous_convolution(x)
        x=self.fc1(x)
        x=self.fc2(x)
        return x

class finetune_VGG(nn.Module):

    def __init__(self):
        super(finetune_VGG,self).__init__()
        self.feature1=nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(256,512,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        )

        self.feature2=nn.Sequential(
            #The value of padding is equal to the value of dilation.
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=2,dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=2,dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        )

    def fine_tune(self):
        vgg = torchvision.models.vgg16(pretrained=True)
        features = list(vgg.features.children())
        block = [self.feature1, self.feature2]
        range = [[0, 22], [24, 29]]
        for idx, layer in enumerate(block):
            for l1, l2 in zip(features[range[idx][0]:range[idx][1]], layer):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data

        print("Fine-tune successful.")

    def forward(self, x):
        feature1 = self.feature1(x)
        feature2 = self.feature2(feature1)
        return feature2

class DeepLabV2_vgg(nn.Module):

    def __init__(self,n_classes,small=True):
        super(DeepLabV2_vgg,self).__init__()
        self.vgg=finetune_VGG()
        self.vgg.fine_tune()

        if small:
            rates = [2, 4, 8, 12]
        else:
            rates = [6, 12, 18, 24]
        self.astrous1=Astrous(2048,n_classes,rates[0])
        self.astrous2=Astrous(2048,n_classes,rates[1])
        self.astrous3=Astrous(2048,n_classes,rates[2])
        self.astrous4=Astrous(2048,n_classes,rates[3])

    def forward(self, x):
        x_vgg=self.vgg(x)
        x1=self.astrous1(x_vgg)
        x2 = self.astrous2(x_vgg)
        x3 = self.astrous3(x_vgg)
        x4 = self.astrous4(x_vgg)

        score=x1+x2+x3+x4
        output=F.interpolate(score,x.size()[2:],mode="bilinear",align_corners=True)

        return output

if __name__=="__main__":
    test_input=torch.rand(1,3,32,32)
    model=DeepLabV2_vgg(n_classes=21)
    output=model(test_input)
    print(output.size())
