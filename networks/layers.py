#coding:utf8
from __future__ import print_function
import torch.nn as nn
from collections import OrderedDict

class depthwise(nn.Module):
    '''
    depthwise convlution
    '''
    def __init__(self, cin, cout, kernel_size=3, stride=1, padding=3, dilation=1, depth=False):
        super(depthwise, self).__init__()
        if depth:
                self.Conv=nn.Sequential(OrderedDict([('conv1_1_depth', nn.Conv3d(cin, cin,
                        kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=cin)),
                        ('conv1_1_point', nn.Conv3d(cin, cout, 1))]))
        else:
            if stride>=1:
                self.Conv=nn.Conv3d(cin, cout, kernel_size=kernel_size, stride=stride,
                                                            padding=padding, dilation=dilation)
            else:
                stride = int(1//stride)
                self.Conv = nn.ConvTranspose3d(cin, cout, kernel_size=kernel_size, stride=stride,
                                                            padding=padding, dilation=dilation)
    def forward(self, x):
        return self.Conv(x)



class SingleConv3D(nn.Module):
    def __init__(self, cin, cout, norm='in', pad=1, depth=False, dilat=1):
        super(SingleConv3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            raise ValueError('please choose the correct normilze method!!!')
        if pad =='same':
            self.padding = dilat
        else:
            self.padding = pad
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1', depthwise(cin, cout, 3, padding=self.padding, depth=depth, dilation=dilat)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU()),
        ]))

    def forward(self, x):
        return self.model(x)

class SingleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, norm='in', kernel_size=3, padding='same', stride=1, dilation=1):
        super(SingleConv2D, self).__init__()
        if norm == 'bn':
            Norm = nn.BatchNorm2d
        elif norm == 'in':
            Norm = nn.InstanceNorm2d
        else:
            raise ValueError('please choose the correct normalize method!!!')
        padding = padding if padding == 'same' else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.norm = Norm(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class SingleResConv3D(nn.Module):
    def __init__(self, cin, cout, norm='in', pad=1, depth=False, dilat=1):
        super(SingleResConv3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            raise ValueError('please choose the correct normilze method!!!')
        self.dilation = [[1, dilat, dilat], [dilat, 1, 1]]
        self.Input = nn.Conv3d(cin, cout, 1)
        self.norm = Norm(cout)
        self.active = nn.ReLU()
        if pad =='same':
            self.padding = dilat
        else:
            self.padding = pad
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1', depthwise(cin, cout, 3, padding=self.padding, depth=depth, dilation=dilat)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU()),
        ]))

    def forward(self, x):

        return self.active(self.norm(self.model(x)+self.Input(x)))