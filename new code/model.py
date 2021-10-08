from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

def downsample(x, num_res):
    '''
    downasample input using maxpooling, applied (num_res-1) times
    :param x: input feature map
    :param num_res: number of resolutions
    :return: list for downsampled features
    '''
    multi_scale_input = [x]
    for idx in range(num_res - 1):
        xx = F.max_pool2d(multi_scale_input[idx], kernel_size=2)
        multi_scale_input.append(xx)
    return multi_scale_input


def string_to_list(x):
    '''
    convert a string to a list of integers
    :param x: a string containing ","
    :return: a list of integers
    '''
    x = x.split(',')
    res = [int(i) for i in x]
    return res

class sConv_multi(nn.Module):
    def __init__(self, in_, out_, kernel_size, max_scales=4, classifier=0):
        '''
        Creates a layer of network by applying convolution over multiple resolutions and summing the results with weight alpha
        Number of filters (and output channels) used is doubled every resolution
        resolution r:
            B: batch size
            k: kernel size

            input: (B, in_, H, W)
            filters: (out_ * 2^r , in_, k, k).
            output: (B, out_ * 2^r, H, W)

        :param in_: number of input channels
        :param out_: number of output channels for first resolution
        :param kernel_size: convolution filter size
        :param max_scales: number of resolutions to use
        :param classifier: 0 or 1, used to distinguish last layer
        '''
        super(sConv_multi, self).__init__()
        self.interp = F.interpolate
        self.max_scales = max_scales
        self.in_ = in_ # input channels
        self.out_ = out_ # output channels
        self.alpha = nn.Parameter(torch.ones(self.max_scales)) # unnormalized weight used in weighted sum of resolution outputs
        self.nalpha = F.softmax(self.alpha, 0).view(-1, 1) # softmax over alpha
        self.conv = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.ModuleList([nn.BatchNorm2d(self.channel_per_res[r], affine=False) for r in range(self.max_scales)]) # batchnorm


    def forward(self, x):
        '''
        input: feature map of size (batch_size, in_, H, W)
        output: feature map of size (batch_size, out_ * (2) ** max_scales, H, W)
        '''
        self.nalpha = F.softmax(self.alpha, 0).view(-1, 1)
        lx = downsample(x, self.max_scales) # a list of inputs for each resolution
        ly = []
        for r in range(self.max_scales):
            y = self.conv(lx[r]) # apply convolution using
            # only number of filters corresponding to that resolution
            y = self.relu(y)
            y = self.bn[r](y)
            y_up = self.interp(y, scale_factor=2 ** r, mode='nearest') # upsample to largest size, using nearest neighbor
            ly.append(y_up)
        y = (torch.stack(ly, 0))
        ys = y.shape
        out = (y * self.nalpha.view(ys[0], 1, 1, 1, 1)).sum(0) # weighted sum using nalpha
        return out


class sConv2d(nn.Module):
    def __init__(self, in_, out_, kernel_size, max_scales=4):
        '''
        Creates a layer of network by applying convolution over multiple resolutions and summing the results with weight alpha

        :param in_: number of input channels
        :param out_: number of output channels
        :param kernel_size: convolution filter size
        :param max_scales: number of resolutions to use
        '''
        super(sConv2d, self).__init__()
        self.interp = F.interpolate
        self.max_scales = max_scales
        self.in_ = in_ # input channels
        self.out_ = out_ # input channels
        self.alpha = nn.Parameter(torch.ones(self.max_scales)) # unnormalized weight used in weighted sum of resolution outputs
        self.nalpha = F.softmax(self.alpha, 0).view(-1, 1) # softmax over alpha

        # convolution
        self.conv = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.ModuleList([nn.BatchNorm2d(self.out_, affine=False) for r in range(self.max_scales)]) # batchnorm


    def forward(self, x):
        '''
        input: feature map of size (batch_size, in_, H, W)
        output: feature map of size (batch_size, out_ , H, W)
        '''
        self.nalpha = F.softmax(self.alpha, 0).view(-1, 1)
        lx = downsample(x, self.max_scales) # a list of inputs for each resolution
        ly = []
        for r in range(self.max_scales):
            y = self.conv(lx[r])
            y = self.relu(y)
            y = self.bn[r](y)
            y_up = self.interp(y, scale_factor=2 ** r, mode='nearest') # upsample to largest size, using nearest neighbor
            ly.append(y_up)
        y = (torch.stack(ly, 0))
        ys = y.shape
        out = (y * self.nalpha.view(ys[0], 1, 1, 1, 1)).sum(0) # weighted sum using nalpha
        return out

class multires_model(nn.Module):
    def __init__(self, ncat=10, channels=16, leng=10, max_scales=4):
        '''
        create CNN by stacking layers
        :param ncat: number of classes
        :param channels: number of channels for first resolution
        :param leng: depth of the network
        :param max_scales: how many scales to use
        '''
        super(multires_model, self).__init__()
        self.leng = leng
        channels = string_to_list(channels)
        listc = [sConv2d(3, channels[0], 3, max_scales)]
        listc += [sConv2d(channels[i] , channels[i+1], 3, max_scales) for i in range(leng - 2)]
        listc += [sConv2d(channels[-1], ncat, 3, max_scales)]
        self.conv = nn.ModuleList(listc)

    def forward(self, x):
        '''
        input: RGB image batch of size (batch_size, 3, H, W)
        output: vector of class probabilities of size (batch_size, ncat)
        '''
        out = x
        for c in range(self.leng):
            out = self.conv[c](out)
        out = F.avg_pool2d(out, kernel_size=out.size()[2:]) # Global average pooling
        out = out.view(out.size(0), -1)
        return out