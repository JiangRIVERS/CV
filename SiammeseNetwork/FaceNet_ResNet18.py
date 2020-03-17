"""
Filename: FaceNet_ResNet18.py

基于ResNet18网络结构的FaceNet

@author: Jiang Rivers
"""
import numpy
import torch
import torch.nn as nn
import torchvision.models as models

# 网络结构
class FaceNet_ResNet18(nn.Module):
    '''
    初期数据量偏少，选用简单ResNet模型进行训练，防止过拟合
    输入[N,C,H,W]
    ResNet18 输出[N,embedding_size]
    '''

    def __init__(self, embedding_size=128, pretrained=False):
        '''
        :params: embedding_size: 图片经过embedding后的size为（N,embedding_size），default=128
        '''
        super(FaceNet_ResNet18, self).__init__()

        self.resnet18 = models.resnet18(pretrained)
        self.resnet18.fc = nn.Linear(512, embedding_size)

    def forward(self, x):

        output = self.resnet18(x)
        return output

# Demo实验
if __name__ == '__main__':
    img = torch.randn((1, 3, 512, 512))
    facenet = FaceNet_ResNet18()
    output = facenet(img)
    print(output.shape)