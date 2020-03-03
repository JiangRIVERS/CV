"""
Filename: Faster_RCNN.py


A Faster RCNN implement based on pytorch.

Reference:https://github.com/jwyang/faster-rcnn.pytorch
@author: Jiang Rivers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class fasterRCNN(nn.Module):
    '''faster RCNN'''

