# DeepLab

DeepLab series connect the output of DCNN with a fully connected Conditional Random Field (CRF), which is shown both qualitatively and quantitatively to improve localization performance.  We will not introduce CRF here.

## V1:
### Title: Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs
A improvementive network of FCN. FCN utilizes a Conv2d with padding=100 to get a dense result, but continuousive conv layers and maxpool layers with stride=2 greatly reduce the resolution which is bad for Objection Segmentation task.
Pool layers with stride=1 will get a dense result and will not reduce the resolution as much as FCN.

<img src="https://img-blog.csdn.net/20161204222427419?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center" width="100%">

However, Pool layers with stride=1 will reduce the Receptive Field as shown above.
The fomulation of RF is :RF=((RF-1)*stride)+kernel_size
DeepLab propose a structure named "atrous convolution(or dilated convolution)" and pool layers with stride=1.

<img src="https://img-blog.csdnimg.cn/20190325194437242.gif" width="50%">



DeepLab fine-tunes VGG16 and it uses bilinear upsampling method.The encoding structure is shown behind.

<img src="https://img-blog.csdn.net/20180529193355787?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDg1OTQzNg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width="30%">

## V2
### Title: DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs

In order to solve the problem of different multi-scale target states in the image(The will be many targets in different scales in the same image and their information may be expressed well in different layers, e.g., target in large scale may be captured well in low semantic layer while target in small scale may be captured well in high semantic layer), DeepLab V2 proposes a structure named
"Atrous Spatial Pyramid Pooling".

<img src="https://img-blog.csdn.net/20180529214929143?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDg1OTQzNg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width="50%">

DeepLab v2 applies ASPP to the output of pool5 layers and concatenates the result of each branch of ASPP.

DeepLab v2 fine-tune two traditional network as follows:

+ VGG
+ ResNet

<img src="https://img-blog.csdn.net/20180529214434336?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDg1OTQzNg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width="50%">

## V3
### Title:Rethinking Atrous Convolution for Semantic Image Segmentation

This paper propose two modules to imrpove DeepLabV2: 
1. Cascade
2. Parallel
+ Cascade
  + Multi-grid Method
  They duplicate several copies of the last ResNet block, denoted as block4, and arrange them in cascade. When output_stride==16 and Multi_Grid=(1,2,4), the three convolutions will have rate=2x(1,2,4)=(2,4,8) in the block4. In the block5, the rate will be 4x(1,2,4)=(4,8,16) and so on.
  

<img src="https://img-blog.csdn.net/20180601130452975?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDg1OTQzNg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width="100%">
+ Parallel
  + promblem with ASPP in DeepLabV2:
  They discover that as the sampling rate becomes larger, the number of valid filter weights (i.e., the weights that are applied to the valid fea- ture region, instead of padded zeros) becomes smaller. This effect is illustrated in Fig. 4 when applying a 3 × 3 filter to a 65 × 65 feature map with different atrous rates. In the extreme case where the rate value is close to the feature map size, the 3 × 3 filter, instead of capturing the whole image context, degenerates to a simple 1 × 1 filter since only the center filter weight is effective.
  <img src="https://img-blog.csdn.net/20180530150928727?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDg1OTQzNg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width="70%">
  + solution:
    + The block4 utilize the Multi_grid method similar to the block4 in Cascade strategy.
    + one 1×1 convolution and three 3 × 3 convolutions with rates = (6, 12, 18) when output stride = 16 (all with 256 filters and batch normalization)
    + Global pooling+bilinearly upsample the feature to the desired spatial dimension. 
    + BN
    <img src="https://img-blog.csdn.net/20180530151007565?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDg1OTQzNg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width="100%">

## V3+
Adding encode-decode construction