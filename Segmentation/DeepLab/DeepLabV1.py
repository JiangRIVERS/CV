"""
Filename:DeepLabV1

In the paper, in order to increase the boundary localization accuracy, they do a
Multi-scale prediction. It is a simple concatenating operation, and
i do not achieve it in this code.

@author Jiang Rivers
"""

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class DeepLabV1(nn.Module):

    def __init__(self,n_classes=21):
        super(DeepLabV1,self).__init__()
        self.n_classes=n_classes
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

        self.classifier=nn.Sequential(
            nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=12,dilation=12),
            nn.Conv2d(1024,1024,kernel_size=1,stride=1,padding=0),
            nn.Conv2d(1024, self.n_classes,kernel_size=1, stride=1, padding=0)
        )

    def fine_tune(self):
        vgg = torchvision.models.vgg16()
        features=list(vgg.features.children())
        block=[self.feature1,self.feature2]
        range=[[0,22],[24,29]]
        for idx,layer in enumerate(block):
            for l1,l2 in zip(features[range[idx][0]:range[idx][1]],layer):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size()==l2.weight.size()
                    assert l1.bias.size()==l2.bias.size()
                    l2.weight.data=l1.weight.data
                    l2.bias.data=l1.bias.data

        print("Fine-tune successful.")

    def forward(self, x):
        feature1=self.feature1(x)
        feature2=self.feature2(feature1)
        score=self.classifier(feature2)
        output= F.interpolate(score,x.size()[2:],mode='bilinear',align_corners=True)
        return output

if __name__=="__main__":
    test_input=torch.rand(1,3,32,32)
    model=DeepLabV1()
    model.fine_tune()
    output=model(test_input)
    print(output.size())







