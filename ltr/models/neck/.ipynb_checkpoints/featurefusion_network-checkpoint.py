# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TransT FeatureFusionNetwork class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import init
import torch



import os
import sys
import inspect



class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x

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

#     def forward(self, x):
#         N, C, H, W = x.size()
#         x = x.view(N, self.group_num, -1)
#         mean = x.mean(dim=2, keepdim=True)
#         std = x.std(dim=2, keepdim=True)
#         x = (x - mean) / (std + self.eps)
#         x = x.view(N, C, H, W)
#         return x * self.weight + self.bias


# class SRU(nn.Module):
#     def __init__(self,
#                  oup_channels: int,
#                  group_num: int = 16,
#                  gate_treshold: float = 0.5,
#                  torch_gn: bool = False
#                  ):
#         super().__init__()

#         self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
#             c_num=oup_channels, group_num=group_num)
#         self.gate_treshold = gate_treshold
#         self.sigomid = nn.Sigmoid()

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

#     def reconstruct(self, x_1, x_2):
#         x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
#         x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
#         return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


# class CRU(nn.Module):
#     '''
#     alpha: 0<alpha<1
#     '''

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

#     def forward(self, x):
#         x = self.SRU(x)
#         x = self.CRU(x)
#         return x
class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0.1, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        # B, C//3, H, W
        B, d, H, W = q.shape
        q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
        k = self.unfold(k).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 2,
                                                                                                        3)  # B,h,N,d,k*k
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 3,
                                                                                                        2)  # B,h,N,k*k,d
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class MultiDilatelocalAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.1, proj_drop=0., kernel_size=3, dilation=[2, 3]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B,C,H,W = x.shape
        # B, H, W, C = x.shape
        # x = x.permute(0, 3, 1, 2)  # B, C, H, W
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        # num_dilation,3,B,C//num_dilation,H,W
        x = x.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2)
        # 先存储每个dilation attention的输出
        dilated_outputs = []
        for i in range(self.num_dilation):
            dilated_output = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])
            dilated_outputs.append(dilated_output)
        x = torch.stack(dilated_outputs, dim=0)  # 将列表转换为新的张量
        # # num_dilation, B, H, W, C//num_dilation
        # for i in range(self.num_dilation):
        #     x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])  # B, H, W,C//num_dilation
        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0,3,1,2).reshape(B,C,H,W)
        return x

class FeatureFusionNetwork(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_featurefusion_layers=3,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.ca1 = MultiDilatelocalAttention(256)
        self.ca2 = MultiDilatelocalAttention(256)
        # self.scc = ScConv(256)
        featurefusion_layer = FeatureFusionLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = Encoder(featurefusion_layer, num_featurefusion_layers)

        decoderCFA_layer = DecoderCFALayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoderCFA_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(decoderCFA_layer, decoderCFA_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_temp, mask_temp, src_search, mask_search, pos_temp, pos_search):
        src_temp=self.ca1(src_temp)
        src_search = self.ca2(src_search)
        # src_temp=self.scc(src_temp)
        # src_search = self.scc(src_search)
        src_temp = src_temp.flatten(2).permute(2, 0, 1)
        pos_temp = pos_temp.flatten(2).permute(2, 0, 1)
        src_search = src_search.flatten(2).permute(2, 0, 1)
        pos_search = pos_search.flatten(2).permute(2, 0, 1)
        mask_temp = mask_temp.flatten(1)
        mask_search = mask_search.flatten(1)

        memory_temp, memory_search = self.encoder(src1=src_temp, src2=src_search,
                                                  src1_key_padding_mask=mask_temp,
                                                  src2_key_padding_mask=mask_search,
                                                  pos_src1=pos_temp,
                                                  pos_src2=pos_search)

        hs = self.decoder(memory_search, memory_temp,
                          tgt_key_padding_mask=mask_search,
                          memory_key_padding_mask=mask_temp,
                          pos_enc=pos_temp, pos_dec=pos_search)

        return hs.unsqueeze(0).transpose(1, 2)


class Decoder(nn.Module):

    def __init__(self, decoderCFA_layer, norm=None):
        super().__init__()
        self.layers = _get_clones(decoderCFA_layer, 1)
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_dec: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos_enc=pos_enc, pos_dec=pos_dec)

        if self.norm is not None:
            output = self.norm(output)

        return output

class Encoder(nn.Module):

    def __init__(self, featurefusion_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(featurefusion_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None):
        output1 = src1
        output2 = src2

        for layer in self.layers:
            output1, output2 = layer(output1, output2, src1_mask=src1_mask,
                                     src2_mask=src2_mask,
                                     src1_key_padding_mask=src1_key_padding_mask,
                                     src2_key_padding_mask=src2_key_padding_mask,
                                     pos_src1=pos_src1, pos_src2=pos_src2)

        return output1, output2


class DecoderCFALayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos_enc: Optional[Tensor] = None,
                     pos_dec: Optional[Tensor] = None):

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, pos_dec),
                                   key=self.with_pos_embed(memory, pos_enc),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_dec: Optional[Tensor] = None):

        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos_enc, pos_dec)

class FeatureFusionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model)

        self.linear21 = nn.Linear(d_model, dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear22 = nn.Linear(dim_feedforward, d_model)

        self.norm11 = nn.LayerNorm(d_model)
        self.norm12 = nn.LayerNorm(d_model)
        self.norm13 = nn.LayerNorm(d_model)
        self.norm21 = nn.LayerNorm(d_model)
        self.norm22 = nn.LayerNorm(d_model)
        self.norm23 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.dropout21 = nn.Dropout(dropout)
        self.dropout22 = nn.Dropout(dropout)
        self.dropout23 = nn.Dropout(dropout)

        self.activation1 = _get_activation_fn(activation)
        self.activation2 = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src1, src2,
                     src1_mask: Optional[Tensor] = None,
                     src2_mask: Optional[Tensor] = None,
                     src1_key_padding_mask: Optional[Tensor] = None,
                     src2_key_padding_mask: Optional[Tensor] = None,
                     pos_src1: Optional[Tensor] = None,
                     pos_src2: Optional[Tensor] = None):
        q1 = k1 = self.with_pos_embed(src1, pos_src1)
        src12 = self.self_attn1(q1, k1, value=src1, attn_mask=src1_mask,
                               key_padding_mask=src1_key_padding_mask)[0]
        src1 = src1 + self.dropout11(src12)
        src1 = self.norm11(src1)

        q2 = k2 = self.with_pos_embed(src2, pos_src2)
        src22 = self.self_attn2(q2, k2, value=src2, attn_mask=src2_mask,
                               key_padding_mask=src2_key_padding_mask)[0]
        src2 = src2 + self.dropout21(src22)
        src2 = self.norm21(src2)


        src12 = self.multihead_attn1(query=self.with_pos_embed(src1, pos_src1),
                                   key=self.with_pos_embed(src2, pos_src2),
                                   value=src2, attn_mask=src2_mask,
                                   key_padding_mask=src2_key_padding_mask)[0]
        src22 = self.multihead_attn2(query=self.with_pos_embed(src2, pos_src2),
                                   key=self.with_pos_embed(src1, pos_src1),
                                   value=src1, attn_mask=src1_mask,
                                   key_padding_mask=src1_key_padding_mask)[0]

        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = self.norm13(src1)

        src2 = src2 + self.dropout22(src22)
        src2 = self.norm22(src2)
        src22 = self.linear22(self.dropout2(self.activation2(self.linear21(src2))))
        src2 = src2 + self.dropout23(src22)
        src2 = self.norm23(src2)

        return src1, src2

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None):

        return self.forward_post(src1, src2, src1_mask, src2_mask,
                                 src1_key_padding_mask, src2_key_padding_mask, pos_src1, pos_src2)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_featurefusion_network(settings):
    return FeatureFusionNetwork(
        d_model=settings.hidden_dim,
        dropout=settings.dropout,
        nhead=settings.nheads,
        dim_feedforward=settings.dim_feedforward,
        num_featurefusion_layers=settings.featurefusion_layers
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
