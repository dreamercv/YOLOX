#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    @staticmethod
    def forward(x):
        return x * (torch.tanh(F.softplus(x)))


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "mish":
        module = Mish()
    elif name == "no":  # 不需要进行激活函数操作
        module = None
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


# 代替3x3卷积
class HSBottleneck(nn.Module):
    def __init__(self, inc, outc, act="relu", dowmsample=False, split_num=5, depthwise=False):
        super(HSBottleneck, self).__init__()
        self.inc = inc
        self.outc = outc
        self.dowmsample = dowmsample
        self.split_num = split_num
        self.act = get_activation(act)

        self.last_inc = self.inc // self.split_num + self.inc % self.split_num  # 52
        self.other_inc = self.inc // self.split_num  # 51

        self.last_outc = self.outc // self.split_num + self.outc % self.split_num  # 104
        self.other_outc = self.outc // self.split_num  # 102

        Conv = DWConv if depthwise else BaseConv

        self.modulelist = nn.ModuleList()
        for split_i in range(self.split_num):
            one_split = nn.ModuleList()
            if split_i == 0:
                if self.dowmsample:
                    one_block_1 = nn.Sequential(Conv(self.other_inc, self.other_outc, ksize=3, stride=2, act="no"),
                                                # conv3x3(self.other_inc, self.other_outc, stride=2),
                                                # nn.BatchNorm2d(self.other_outc)
                                                )
                else:
                    if inc != outc:
                        one_block_1 = nn.Sequential(Conv(self.other_inc, self.other_outc, ksize=1, stride=1, act="no"),
                                                    # conv3x3(self.other_inc, self.other_outc, stride=1),
                                                    # nn.BatchNorm2d(self.other_outc)
                                                    )
                    else:
                        one_block_1 = None
                one_split.append(one_block_1)
            elif split_i == 1:
                if self.dowmsample:
                    one_block_2 = nn.Sequential(Conv(self.other_inc, self.other_outc * 2, ksize=3, stride=2, act="no"),
                                                # conv3x3(self.other_inc, self.other_outc * 2, stride=2),
                                                # nn.BatchNorm2d(self.other_outc * 2)
                                                )
                else:
                    one_block_2 = nn.Sequential(Conv(self.other_inc, self.other_outc * 2, ksize=3, stride=1, act="no"),
                                                # conv3x3(self.other_inc, self.other_outc * 2),
                                                # nn.BatchNorm2d(self.other_outc * 2)
                                                )
                one_split.append(one_block_2)
            elif split_i == self.split_num - 1:
                if self.dowmsample:
                    one_block_last_1 = nn.Sequential(Conv(self.last_inc, self.last_inc, ksize=3, stride=2, act="no"),
                                                     # conv3x3(self.last_inc, self.last_inc, stride=2),
                                                     # nn.BatchNorm2d(self.last_inc)
                                                     )
                else:
                    if inc != outc:
                        one_block_last_1 = nn.Sequential(
                            Conv(self.last_inc, self.last_inc, ksize=1, stride=1, act="no"),
                            # conv3x3(self.last_inc, self.last_inc, stride=1),
                            # nn.BatchNorm2d(self.last_inc)
                        )
                    else:
                        one_block_last_1 = None
                one_block_last_2 = nn.Sequential(conv1x1(self.last_inc + self.other_outc, self.last_outc),
                                                 nn.BatchNorm2d(self.last_outc))
                one_split.append(one_block_last_1), one_split.append(one_block_last_2)
            else:
                if self.dowmsample:
                    one_block_middle_1 = nn.Sequential(
                        Conv(self.other_inc, self.other_outc, ksize=3, stride=2, act="no"),
                        # conv3x3(self.other_inc, self.other_outc, stride=2),
                        # nn.BatchNorm2d(self.other_outc)
                    )
                else:
                    if inc != outc:
                        one_block_middle_1 = nn.Sequential(
                            Conv(self.other_inc, self.other_outc, ksize=1, stride=1, act="no"),
                            # conv3x3(self.other_inc, self.other_outc, stride=1),
                            # nn.BatchNorm2d(self.other_outc)
                        )
                    else:
                        one_block_middle_1 = None
                one_block_middle_2 = nn.Sequential(
                    Conv(self.other_outc * 2, self.other_outc * 2, ksize=3, stride=1, act="no"),
                    # conv3x3(self.other_outc * 2, self.other_outc * 2),
                    # nn.BatchNorm2d(self.other_outc * 2)
                )
                one_split.append(one_block_middle_1), one_split.append(one_block_middle_2)
            self.modulelist.append(one_split)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x[:, :self.other_inc, :, :]
        need_concat = x

        for i in range(self.split_num):
            if i == 0:
                if self.modulelist[i][0]:

                    out = self.modulelist[i][0](out)
                else:
                    continue
                # print(out.shape)
            elif i == 1:
                split_1 = self.modulelist[i][0](x[:, self.other_inc * i:self.other_inc * (i + 1), :, :])
                split_1_1, need_concat = split_1[:, :self.other_outc, :, :], split_1[:, self.other_outc:, :, :]
                out = torch.cat((out, split_1_1), dim=1)
            else:
                if i == self.split_num - 1:
                    if self.modulelist[i][0]:
                        split_1 = torch.cat((self.modulelist[i][0](x[:, self.other_inc * i:, :, :]), need_concat),
                                            dim=1)
                    else:
                        split_1 = torch.cat((x[:, self.other_inc * i:, :, :], need_concat), dim=1)
                    split_2 = self.modulelist[i][1](split_1)
                    out = torch.cat((out, split_2), dim=1)
                else:
                    if self.modulelist[i][0]:
                        split_1 = torch.cat((self.modulelist[i][0](
                            x[:, self.other_inc * i:self.other_inc * (i + 1), :, :]), need_concat), dim=1)
                    else:
                        split_1 = torch.cat((
                            x[:, self.other_inc * i:self.other_inc * (i + 1), :, :], need_concat), dim=1)
                    split_2 = self.modulelist[i][1](split_1)
                    split_2_1, need_concat = split_2[:, self.other_outc:, :, :], split_2[:, self.other_outc:, :, :]
                    out = torch.cat((out, need_concat), dim=1)

        return self.act(out)


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)

        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        if self.act is None:
            return self.bn(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        if self.act is None:
            return self.conv(x)
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
            self,
            in_channels,
            out_channels,
            shortcut=True,
            expansion=0.5,
            depthwise=False,
            act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
            self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
            self,
            in_channels,
            out_channels,
            n=1,
            shortcut=True,
            expansion=0.5,
            depthwise=False,
            act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


if __name__ == '__main__':
    '''
    HSBottleneck(256, 256, depthwise=False, dowmsample=False)
    HSnet 总参数数量和：240162
    Basenet 总参数数量和：590336
    DWnet 总参数数量和：68864
    
    HSBottleneck(256, 512, depthwise=False, dowmsample=False)
    HSnet 总参数数量和：879198
    Basenet 总参数数量和：1180672
    DWnet 总参数数量和：134912
    
    HSnet = HSBottleneck(256, 256, depthwise=False, dowmsample=True)
    HSnet 总参数数量和：335135
    Basenet 总参数数量和：590336
    DWnet 总参数数量和：68864
    
    HSnet = HSBottleneck(256, 512, depthwise=False, dowmsample=True)
    HSnet 总参数数量和：1025678
    Basenet 总参数数量和：1180672
    DWnet 总参数数量和：134912
    
    
    
    
    '''
    HSnet = HSBottleneck(256, 256, depthwise=True, dowmsample=True)
    Basenet = BaseConv(256,256,3,2)
    DWnet = DWConv(256,256,3,2)
    # print(HSnet)
    img = torch.ones([1, 256, 288, 288])
    out = HSnet(img)
    # print(out.shape)

    params = list(HSnet.parameters())
    k = 0
    for i in params:
        l = 1
        # print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        # print("该层参数和：" + str(l))
        k = k + l
    print("HSnet 总参数数量和：" + str(k))

    params = list(Basenet.parameters())
    k = 0
    for i in params:
        l = 1
        # print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        # print("该层参数和：" + str(l))
        k = k + l
    print("Basenet 总参数数量和：" + str(k))

    params = list(DWnet.parameters())
    k = 0
    for i in params:
        l = 1
        # print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        # print("该层参数和：" + str(l))
        k = k + l
    print("DWnet 总参数数量和：" + str(k))



