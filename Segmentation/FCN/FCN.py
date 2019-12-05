"""
Filename:FCN

This FCN's pytorch implementation referring to the FCN paper source code
:http://fcn.berkeleyvision.org.
This is only document for learning the structure of FCN.
For entire code of a project based on FCN, please check
https://github.com/meetshah1995/pytorch-semseg,
this document is only a practice for me after reading his code.
I only apply the fine-tune part of vgg16 for the network and
the rest of the layers,5 layers in total,
are not be initialized by pytorch automatically.

@author Jiang Rivers
"""
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

vgg=torchvision.models.vgg16()

class FCN8s(nn.Module):
    """Implement of FCN8s"""

    def __init__(self,n_classes=21,learned_billinear=True,finetune_model=vgg):
        """
        Parameters:
            n_classes:The number of the classes of the classifier layer.
            learned_billinear:If True, the upsampling layers will take ConvTranspose2D method.
                              Otherwise, the upsamling layers will take F.upsample(filling zero) method.
        """
        super(FCN8s,self).__init__()
        self.n_classes=n_classes
        self.learned_billinear=learned_billinear
        self.finetune_model=finetune_model

        self.block1=nn.Sequential(
            nn.Conv2d(3,64,3,stride=1,padding=100),
            #In order to get a dense feature map,the padding is set to 100.
            #But in this way,there may be some noise in the output.
            #DeepLab has a good improvement.
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,stride=2,padding=0,dilation=1,ceil_mode=False)
            #In Pytorch official code of vgg16,the ceil_mode=False.So in order to meet the
            #Pytorch official code, i set the ceil_mode of FCN8s to False.
            #In https://github.com/meetshah1995/pytorch-semseg the author set the ceil_mode
            #to True.
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.classifier=nn.Sequential(
            nn.Conv2d(512,4096,7,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(4096,4096,1,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(4096,self.n_classes,1,stride=1,padding=0)
        )

        self.convpool4=nn.Conv2d(512,n_classes,1)
        self.convpool3=nn.Conv2d(256,n_classes,1)

        if self.learned_billinear:
            self.upconv7=nn.ConvTranspose2d(self.n_classes,self.n_classes,kernel_size=4,
                                            stride=2,bias=False)
            self.upconv4=nn.ConvTranspose2d(self.n_classes,self.n_classes,kernel_size=4,
                                            stride=2,bias=False)
            self.upconv3=nn.ConvTranspose2d(self.n_classes,self.n_classes,kernel_size=16,
                                           stride=8,bias=False)

    def forward(self, x):
        conv1=self.block1(x)
        conv2=self.block2(conv1)
        conv3=self.block3(conv2)
        conv4=self.block4(conv3)
        conv5=self.block5(conv4)
        conv7=self.classifier(conv5)

        if self.learned_billinear:
            upconv7=self.upconv7(conv7)
            mergecon4_7=self.convpool4(conv4[:,:,5:5+upconv7.size()[2],5:5+upconv7.size()[3]])+upconv7

            upconv4=self.upconv4(mergecon4_7)
            mergeocn3_4_7=self.convpool3(conv3[:,:,9:9+upconv4.size()[2],9:9+upconv4.size()[3]])+upconv4

            output=self.upconv3(mergeocn3_4_7)[:,:,31:31+x.size()[2],31:31+x.size()[3]]
            return output

        convpool4=self.convpool4(conv4)
        convpool3=self.convpool3(conv3)
        upconv7=F.upsample(conv7,convpool4.size()[2:])
        upconv4=F.upsample(upconv7+convpool4,convpool3.size()[2:])
        output=F.upsample(upconv4+convpool3,x.size()[2:])

        return output

    def fine_tune(self,copy_fc8=True):
        blocks = [
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5,
        ]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(self.finetune_model.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0] : ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data

        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = self.finetune_model.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.reshape(l2.weight.size())
            l2.bias.data = l1.bias.data.reshape(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]

        if copy_fc8:
            l1 = self.finetune_model.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].reshape(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]

        print("load parameters successfully.")

if __name__=="__main__":
    test_input=torch.rand(1,3,32,32)
    model=FCN8s()
    model.fine_tune()
    output=model(test_input)
    print(output.size())




