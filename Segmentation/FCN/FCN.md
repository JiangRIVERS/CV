# FCN

## Key
+ Building "fully convolutional" network discarding fc layers
  + take input of arbitrary size and produce correspondingly-sized output with efficient inference and learning
+ Fine-tuning models, e.g., AlexNet, VGG and GoogLeNet
+ Define a skip architecture
+ Utilizing a Conv2D with padding==100 to get a dense result.

## Structure
<img src="https://img-blog.csdnimg.cn/20190402190620569.png" width="100%">


2x conv7 is the upsampling result of conv7 in order to fit the size of pool4.Operation:add 2xconv7 and pool4

## Note

+ Semantic segmentation faces an inherent tension be- tween semantics and location: global information resolves what while local information resolves where. 
Reference:https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/fcn.py
original code:https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/siftflow-fcn8s/net.py

