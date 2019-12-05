"""
Filename: DeepLabV2_res.py

A DeepLabV2 implementation based on ResNet101.

@author Jiang Rivers
"""
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class Astrous_Bottleneck(nn.Module):
    expansion=4

    def __init__(self,inplanes,planes,stride=1,rate=1,downsample=False):
        super(Astrous_Bottleneck,self).__init__()
        self.stride=stride
        self.downsample=downsample
        self.conv1=nn.Conv2d(inplanes,planes,kernel_size=1,bias=False)
        self.bn1=nn.BatchNorm2d(planes)

        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=rate,dilation=rate,bias=False)
        self.bn2=nn.BatchNorm2d(planes)

        self.conv3=nn.Conv2d(planes,Astrous_Bottleneck.expansion*planes,kernel_size=1,bias=False)
        self.bn3=nn.BatchNorm2d(Astrous_Bottleneck.expansion*planes)
        self.relu=nn.ReLU(inplace=True)
        if self.downsample:
            self.downsample=nn.Sequential(
                nn.Conv2d(inplanes,planes*Astrous_Bottleneck.expansion,kernel_size=1,stride=stride, bias=False),
                nn.BatchNorm2d(planes*Astrous_Bottleneck.expansion)
            )


    def forward(self, x):

        residual=x

        output=self.conv1(x)
        output=self.bn1(output)
        output=self.relu(output)

        output=self.conv2(output)
        output=self.bn2(output)
        output=self.relu(output)

        output=self.conv3(output)
        output=self.bn3(output)

        if self.downsample:
            residual=self.downsample(x)

        output+=residual
        output=self.relu(output)

        return output

class Resnet_features(nn.Module):

    def __init__(self,block,pretrained=False):
        super(Resnet_features,self).__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=self._make_layer(64,64,block[0],1,1)
        self.layer2=self._make_layer(256,128,block[1],2,1)
        self.layer3=self._make_layer(512,256,block[2],1,2)
        self.layer4=self._make_layer(1024,512,block[3],1,4)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

        if pretrained:
            resnet=torchvision.models.resnet101(pretrained=True)
            self.state_dict().update({k:v for k,v in resnet.state_dict().items() if \
                                      k in self.state_dict()})
            self.load_state_dict(self.state_dict())
            print("Pretraining complete.")


    def _make_layer(self,inplanes,planes,block,stride,rate):
        layers=[]
        layers.append(Astrous_Bottleneck(inplanes,planes,stride,rate,downsample=True))
        for i in range(1,block):
            layers.append((Astrous_Bottleneck(Astrous_Bottleneck.expansion * planes, planes, stride=1,rate=rate)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        return x

class Astrous(nn.Module):

    def __init__(self,planes,n_classes,rate):
        super(Astrous,self).__init__()
        self.atrous_convolution = nn.Conv2d(2048, planes, kernel_size=3,
                                            stride=1, padding=rate, dilation=rate)
        self.fc1=nn.Conv2d(planes,planes,kernel_size=1,stride=1,padding=0)
        self.fc2 = nn.Conv2d(planes, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x=self.atrous_convolution(x)
        x=self.fc1(x)
        x=self.fc2(x)
        return x

class DeepLabV2_res(nn.Module):

    def __init__(self,planes=2048,num_classes=1000,small=True):
        super(DeepLabV2_res,self).__init__()
        block=[3,4,23,3]
        if small:
            rates=[2,4,8,12]
        else:
            rates=[6,12,18,24]

        self.Resnet_features=Resnet_features(block,pretrained=True)
        self.Astrous1=Astrous(planes,num_classes,rates[0])
        self.Astrous2 = Astrous(planes, num_classes, rates[1])
        self.Astrous3 = Astrous(planes, num_classes, rates[2])
        self.Astrous4 = Astrous(planes, num_classes, rates[3])

    def forward(self, x):
        x_res=self.Resnet_features(x)
        x1=self.Astrous1(x_res)
        x2=self.Astrous2(x_res)
        x3=self.Astrous3(x_res)
        x4=self.Astrous4(x_res)

        score=x1+x2+x3+x4
        output=F.interpolate(score,x.size()[2:],mode="bilinear",align_corners=True)

        return output

if __name__=="__main__":
    test_input=torch.rand(1,3,224,224)
    model=DeepLabV2_res()
    output=model(test_input)
    print(output.size())





