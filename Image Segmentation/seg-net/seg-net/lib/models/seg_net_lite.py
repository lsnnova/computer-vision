from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict

logger = logging.getLogger(__name__)


class SegNetLite(nn.Module):

    def __init__(self, kernel_sizes=[3, 3, 3, 3], down_filter_sizes=[32, 64, 128, 256],
            up_filter_sizes=[128, 64, 32, 32], conv_paddings=[1, 1, 1, 1],
            pooling_kernel_sizes=[2, 2, 2, 2], pooling_strides=[2, 2, 2, 2], **kwargs):
        """Initialize SegNet Module

        Args:
            kernel_sizes (list of ints): kernel sizes for each convolutional layer in downsample/upsample path.
            down_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the downsample path.
            up_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the upsample path.
            conv_paddings (list of ints): paddings for each convolutional layer in downsample/upsample path.
            pooling_kernel_sizes (list of ints): kernel sizes for each max-pooling layer and its max-unpooling layer.
            pooling_strides (list of ints): strides for each max-pooling layer and its max-unpooling layer.
        """
        super(SegNetLite, self).__init__()
        self.num_down_layers = len(kernel_sizes)
        self.num_up_layers = len(kernel_sizes)

        input_size = 3 # initial number of input channels
        # Construct downsampling layers.
        # As mentioned in the assignment, blocks of the downsampling path should have the
        # following output dimension (igoring batch dimension):
        # 3 x 64 x 64 (input) -> 32 x 32 x 32 -> 64 x 16 x 16 -> 128 x 8 x 8 -> 256 x 4 x 4
        # each block should consist of: Conv2d->BatchNorm2d->ReLU->MaxPool2d
        layers_conv_down = []
        layers_bn_down = []
        layers_pooling = []
        #raise NotImplementedError('Downsampling layers are not implemented!')
        in_channels = input_size
        for i in range(self.num_down_layers):
            out_channels = down_filter_sizes[i]
            layers_conv_down.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes[i], padding=conv_paddings[i]))
            layers_bn_down.append(nn.BatchNorm2d(out_channels))
            layers_pooling.append(nn.MaxPool2d(kernel_size=pooling_kernel_sizes[i], stride=pooling_strides[i], return_indices=True))
            in_channels = out_channels

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # package can track gradients and update parameters of these layers
        self.layers_conv_down = nn.ModuleList(layers_conv_down)
        self.layers_bn_down = nn.ModuleList(layers_bn_down)
        self.layers_pooling = nn.ModuleList(layers_pooling)

        # Construct upsampling layers
        # As mentioned in the assignment, blocks of the upsampling path should have the
        # following output dimension (igoring batch dimension):
        # 256 x 4 x 4 (input) -> 128 x 8 x 8 -> 64 x 16 x 16 -> 32 x 32 x 32 -> 32 x 64 x 64
        # each block should consist of: MaxUnpool2d->Conv2d->BatchNorm2d->ReLU
        layers_conv_up = []
        layers_bn_up = []
        layers_unpooling = []
        #raise NotImplementedError('Upsampling layers are not implemented!')
        in_channels = down_filter_sizes[-1]
        for i in range(self.num_up_layers):
            out_channels = up_filter_sizes[i]
            layers_unpooling.append(nn.MaxUnpool2d(kernel_size=pooling_kernel_sizes[i], stride=pooling_strides[i]))
            layers_conv_up.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes[i], padding=conv_paddings[i]))
            layers_bn_up.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # can track gradients and update parameters of these layers
        self.layers_conv_up = nn.ModuleList(layers_conv_up)
        self.layers_bn_up = nn.ModuleList(layers_bn_up)
        self.layers_unpooling = nn.ModuleList(layers_unpooling)

        self.relu = nn.ReLU(True)

        # Implement a final 1x1 convolution to to get the logits of 11 classes (background + 10 digits)
        #raise NotImplementedError('Final convolution layer is not implemented!')
        self.final_conv = nn.Conv2d(up_filter_sizes[-1], 11, kernel_size=1)

    def forward(self, x):
        #raise NotImplementedError('Forward function not implemented!')
        indices_list = []
        sizes = []

        # Downsampling path
        for conv, bn, pool in zip(self.layers_conv_down, self.layers_bn_down, self.layers_pooling):
            x = self.relu(bn(conv(x)))
            sizes.append(x.size())
            x, indices = pool(x)
            indices_list.append(indices)

        # Upsampling path
        for unpool, conv, bn, size, indices in zip(self.layers_unpooling, self.layers_conv_up, self.layers_bn_up,
                                                   reversed(sizes), reversed(indices_list)):
            x = unpool(x, indices, output_size=size)
            x = self.relu(bn(conv(x)))

        # Final classification layer
        x = self.final_conv(x)
        return x

def get_seg_net(**kwargs):

    model = SegNetLite(**kwargs)

    return model
