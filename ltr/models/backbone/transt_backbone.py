# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List
import ltr.models.backbone as backbones

from util.misc import NestedTensor

from ltr.models.neck.position_encoding import build_position_encoding

import torch
import torch.nn.functional as F
import torch.nn as nn


# class GroupBatchnorm2d(nn.Module):
#     def __init__(self, c_num: int,
#                  group_num: int = 16,
#                  eps: float = 1e-10
#                  ):
#         super(GroupBatchnorm2d, self).__init__()
#         assert c_num >= group_num
#         self.group_num = group_num
#         self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
#         self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
#         self.eps = eps
#
#     def forward(self, x):
#         N, C, H, W = x.size()
#         x = x.view(N, self.group_num, -1)
#         mean = x.mean(dim=2, keepdim=True)
#         std = x.std(dim=2, keepdim=True)
#         x = (x - mean) / (std + self.eps)
#         x = x.view(N, C, H, W)
#         return x * self.weight + self.bias
#
#
# class SRU(nn.Module):
#     def __init__(self,
#                  oup_channels: int,
#                  group_num: int = 16,
#                  gate_treshold: float = 0.5,
#                  torch_gn: bool = False
#                  ):
#         super().__init__()
#
#         self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
#             c_num=oup_channels, group_num=group_num)
#         self.gate_treshold = gate_treshold
#         self.sigomid = nn.Sigmoid()
#
#     def forward(self, x):
#         gn_x = self.gn(x)
#         w_gamma = self.gn.weight / torch.sum(self.gn.weight)
#         w_gamma = w_gamma.view(1, -1, 1, 1)
#         reweigts = self.sigomid(gn_x * w_gamma)
#         # Gate
#         info_mask = reweigts >= self.gate_treshold
#         noninfo_mask = reweigts < self.gate_treshold
#         x_1 = info_mask * gn_x
#         x_2 = noninfo_mask * gn_x
#         x = self.reconstruct(x_1, x_2)
#         return x
#
#     def reconstruct(self, x_1, x_2):
#         x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
#         x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
#         return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)
#
#
# class CRU(nn.Module):
#     '''
#     alpha: 0<alpha<1
#     '''
#
#     def __init__(self,
#                  op_channel: int,
#                  alpha: float = 1 / 2,
#                  squeeze_radio: int = 2,
#                  group_size: int = 2,
#                  group_kernel_size: int = 3,
#                  ):
#         super().__init__()
#         self.up_channel = up_channel = int(alpha * op_channel)
#         self.low_channel = low_channel = op_channel - up_channel
#         self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
#         self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
#         # up
#         self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
#                              padding=group_kernel_size // 2, groups=group_size)
#         self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
#         # low
#         self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
#                               bias=False)
#         self.advavg = nn.AdaptiveAvgPool2d(1)
#
#     def forward(self, x):
#         # Split
#         up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
#         up, low = self.squeeze1(up), self.squeeze2(low)
#         # Transform
#         Y1 = self.GWC(up) + self.PWC1(up)
#         Y2 = torch.cat([self.PWC2(low), low], dim=1)
#         # Fuse
#         out = torch.cat([Y1, Y2], dim=1)
#         out = F.softmax(self.advavg(out), dim=1) * out
#         out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
#         return out1 + out2
#
#
# class ScConv(nn.Module):
#     def __init__(self,
#                  op_channel: int,
#                  group_num: int = 4,
#                  gate_treshold: float = 0.5,
#                  alpha: float = 1 / 2,
#                  squeeze_radio: int = 2,
#                  group_size: int = 2,
#                  group_kernel_size: int = 3,
#                  ):
#         super().__init__()
#         self.SRU = SRU(op_channel,
#                        group_num=group_num,
#                        gate_treshold=gate_treshold)
#         self.CRU = CRU(op_channel,
#                        alpha=alpha,
#                        squeeze_radio=squeeze_radio,
#                        group_size=group_size,
#                        group_kernel_size=group_kernel_size)
#
#     def forward(self, x):
#         x = self.SRU(x)
#         x = self.CRU(x)
#         return x
class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, num_channels: int):
        super().__init__()
        self.body = backbone
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self,
                 output_layers,
                 pretrained,
                 frozen_layers):
        backbone = backbones.resnet50(output_layers=output_layers, pretrained=pretrained,
                                      frozen_layers=frozen_layers)
        # backbone =ScConv(3)
        num_channels = 1024
        super().__init__(backbone, num_channels)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(settings, backbone_pretrained=True, frozen_backbone_layers=()):
    position_embedding = build_position_encoding(settings)
    backbone = Backbone(output_layers=['layer3'], pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
