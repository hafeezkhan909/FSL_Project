from __future__ import print_function
import torch as t
import torch.nn as nn
from networks.layers import SingleConv2D
import math
import numpy as np

class PRNet(t.nn.Module):
    def __init__(self,  inc=1, patch_size=1, n_classes=1, base_chns=12, droprate=0, norm='in', dilation=1):
        super(PRNet, self).__init__()
        self.model_name = "seg"
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsample = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(droprate)

        self.conv0_1 = SingleConv2D(inc, base_chns, norm=norm, dilation=dilation, padding='same')
        self.conv0_2 = SingleConv2D(base_chns, base_chns, norm=norm, dilation=dilation, padding='same')

        self.conv1_1 = SingleConv2D(base_chns, base_chns, norm=norm, dilation=dilation, padding='same')
        self.conv1_2 = SingleConv2D(base_chns, 2*base_chns, norm=norm, dilation=dilation, padding='same')

        self.conv2_1 = SingleConv2D(2*base_chns, 2*base_chns, norm=norm, dilation=dilation, padding='same')
        self.conv2_2 = SingleConv2D(2 * base_chns, 4 * base_chns, norm=norm, dilation=dilation, padding='same')

        self.conv3_1 = SingleConv2D(4*base_chns, 4*base_chns, norm=norm, dilation=dilation, padding='same')
        self.conv3_2 = SingleConv2D(4 * base_chns, 8 * base_chns, norm=norm, dilation=dilation, padding='same')

        self.conv4_1 = SingleConv2D(8*base_chns, 8*base_chns, norm=norm, dilation=math.ceil(dilation/2), padding='same')
        self.conv4_2 = SingleConv2D(8 * base_chns, 16 * base_chns, norm=norm, dilation=math.ceil(dilation/2), padding='same')

        self.conv5_1 = SingleConv2D(24*base_chns, 8*base_chns, norm=norm, dilation=dilation, padding='same')
        self.conv5_2 = SingleConv2D(8 * base_chns, 8 * base_chns, norm=norm, dilation=dilation, padding='same')

        self.conv6_1 = SingleConv2D(12*base_chns, 4*base_chns, norm=norm, dilation=dilation, padding='same')
        self.conv6_2 = SingleConv2D(4 * base_chns, 4 * base_chns, norm=norm, dilation=dilation, padding='same')

        self.conv7_1 = SingleConv2D(6*base_chns, 2*base_chns, norm=norm, dilation=dilation, padding='same')
        self.conv7_2 = SingleConv2D(2 * base_chns,  base_chns, norm=norm, dilation=dilation, padding='same')

        self.conv8_1 = SingleConv2D(2*base_chns, base_chns, norm=norm, dilation=dilation, padding='same')
        self.conv8_2 = SingleConv2D(base_chns, base_chns, norm=norm, dilation=dilation, padding='same')

        self.classification = nn.Sequential(
            nn.Conv2d(in_channels=base_chns, out_channels=n_classes, kernel_size=1),
        )
        # fc_inc = int(np.asarray(patch_size).prod()/4096)*16*base_chns
        fc_inc = int(np.asarray(patch_size).prod())
        self.fc1 = nn.Linear(2304, 8 * base_chns)
        self.fc2 = nn.Linear(8 * base_chns, 4 * base_chns)
        self.fc3 = nn.Linear(4 * base_chns, 2)

    def forward(self, x, out_feature=True):
        out = self.conv0_1(x)
        # print('conv0_1 output: ', out.shape)
        conv0 = self.conv0_2(out)
        # print('conv0_2 output: ', conv0.shape)
        out = self.downsample(conv0)
        # print('conv0_2 downsampled output: ', out.shape)
        out = self.conv1_1(out)
        # print('conv1_1 output: ', out.shape)
        conv1 = self.conv1_2(out)
        # print('conv1_2 output: ', conv1.shape)
        out = self.downsample(conv1)  # 1/2
        # print('conv1_2 downsampled output: ', out.shape)
        out = self.conv2_1(out)
        # print('conv2_1 output: ', out.shape)
        conv2 = self.conv2_2(out)  #
        # print('conv2_2 output: ', conv2.shape)
        out = self.downsample(conv2)  # 1/4
        # print('conv2_2 downsampled output: ', out.shape)
        out = self.conv3_1(out)
        # print('conv3_1 output: ', out.shape)
        conv3 = self.conv3_2(out)  #
        # print('conv3_2 output: ', conv3.shape)
        out = self.downsample(conv3)  # 1/8
        # print('conv3_2 downsampled output: ', out.shape)
        out = self.conv4_1(out)
        # print('conv4_1 output: ', out.shape)
        out = self.conv4_2(out)
        # print('conv4_2 output: ', out.shape)
        #out = self.drop(out)

        fc_out = out.view(out.shape[0],-1)
        # print("fc_out output", fc_out.shape)
        fc_out = self.fc1(fc_out)
        # print('fc1 output: ', fc_out.shape)
        fc_out = self.fc2(fc_out)
        # print('fc2 output: ', fc_out.shape)
        fc_out = self.fc3(fc_out)
        # print('fc3 output: ', fc_out.shape)

        up5 = self.upsample(out)  # 1/4
        # print('up5 output: ', up5.shape)
        out = t.cat((up5, conv3), 1)
        # print('concatenating up5 and conv3_2 output: ', out.shape)
        out = self.conv5_1(out)
        # print('conv5_1 output: ', out.shape)
        conv5 = self.conv5_2(out)
        # print('conv5_2 output: ', conv5.shape)
        center_feature5 = conv5[:, :, conv5.shape[2]//2, conv5.shape[3]//2]

        up6 = self.upsample(conv5)  # 1/2
        # print('up6 output: ', up6.shape)
        out = t.cat((up6, conv2), 1)
        # print('concatenating up6 and conv2_2 output: ', out.shape)
        out = self.conv6_1(out)
        # print('conv6_1 output: ', out.shape)
        conv6 = self.conv6_2(out)
        # print('conv6_2 output: ', conv6.shape)
        center_feature6 = conv6[:, :, conv6.shape[2]//2, conv6.shape[3]//2]

        up7 = self.upsample(out)
        # print('up7 output: ', up7.shape)
        out = t.cat((up7, conv1), 1)
        # print('concatenating up7 and conv1_2 output: ', out.shape)
        out = self.conv7_1(out)
        # print('conv7_1 output: ', out.shape)
        conv7 = self.conv7_2(out)
        # print('conv7_2 output: ', conv7.shape)
        center_feature7 = conv7[:, :, conv7.shape[2]//2, conv7.shape[3]//2]

        up8 = self.upsample(conv7)
        # print('up8 output: ', up8.shape)
        out = t.cat((up8, conv0), 1)
        # print('concatenating up8 and conv0_2 output: ', out.shape)
        out = self.conv8_1(out)
        # print('conv8_1 output: ', out.shape)
        conv8 = self.conv8_2(out)
        # print('conv8_2 output: ', conv8.shape)
        center_feature8 = conv8[:, :, conv8.shape[2]//2, conv8.shape[3]//2]

        out = self.classification(out)
        # print('classification output: ', out.shape)
        dic = {'fc_position': fc_out, 'ae': out}
        # print(dic)
        if out_feature:
            dic['center_feature5'] = center_feature5
            dic['center_feature6'] = center_feature6
            dic['center_feature7'] = center_feature7
            dic['center_feature8'] = center_feature8
            dic['fine_feature'] = conv8
        return dic