# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        """
        src:         [bs, c, h, w]  图片输入backbone + 1x1 conv之后的特征图
        mask:        [bs, h, w]     用于记录特征图中哪些地方是填充的.原图部分值为False, 填充部分值为True
        query_embed: [n, c]         类似于传统目标检测里面的anchor,这里设置了100个需要预测的目标
        pos_embed:   [bs, c, h, w]  位置编码
        """
        # TODO: 实现 Transformer 模型的前向传播逻辑
        # 1. 将输入展平，将形状从 (bs, c, h, w) 变为 (hw, bs, c)
        bs, c, h, w = src.shape
        x = src.flatten(2).permute(2, 0, 1)

        # 2. 初始化需要预测的目标 query embedding [这块没有太理解]
        ## pos_embed  : (bs, c, h, w) -> (hw, bs, c)
        ## query_embed: (n, c)        -> (n, bs, c)
        ## mask       : (bs, h, w)    -> (bs, hw)
        ## TODO: 根据后续代码注释，实际为(bs, hw)，那为什么不设计转换为 (hw， bs)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        ### unsqueeze(1): 在1维度上插入一个维度
        ### repeat: 对某个维度进行复制指定次数
        query_embed = query_embed.unsqueeze(1).repeat(1, src.shape[0], 1)
        mask = mask.flatten(1)

        # 3. 使用编码器处理输入序列，得到具有全局相关性（增强后）的特征表示
        memory = self.encoder(x, src_key_padding_mask=mask, pos=pos_embed)
        # print("memory.shape: ", memory.shape)

        # 4. 使用解码器处理目标张量和编码器的输出，得到output embedding
        tgt = torch.zeros_like(query_embed)
        decoder = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)

        # print("decoder.shape: ", decoder.shape)

        # 5. 对输出结果进行形状变换，并返回
        ## decoder输出 [1, n, bs, c] -> [1, bs, n, c]
        decoder_result = decoder.transpose(1, 2)
        # print("decoder_result.shape: ", decoder_result.shape)
        ## encoder输出 [hw, bs, c] -> [bs, c, hw] -> [bs, c, h, w]
        encoder_result = memory.permute(1, 2, 0).view(bs, c, h, w)
        # print("encoder_result.shape: ", encoder_result.shape)

        return decoder_result, encoder_result


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # TODO: 实现 Transformer 编码器的前向传播逻辑
        # 1. 遍历$num_layers层TransformerEncoderLayer
        for i, layer in enumerate(self.layers):
            src = layer(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        # 2. layer norm
        if self.norm is not None:
            src = self.norm(src)

        # 3. 得到最终编码器的输出
        return src


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # 是否返回中间层 默认True  因为DETR默认6个Decoder都会返回结果，一起加入损失计算的
        # 每一层Decoder都是逐层解析，逐层加强的，所以前面层的解析效果对后面层的解析是有意义的，所以作者把前面5层的输出也加入损失计算
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """
        tgt                     : [n,  bs, c]      需要预测的目标query embedding
        memory                  : [hw, bs, c]      Encoder输出
        tgt_mask                : None
        tgt_key_padding_mask    : None
        memory_key_padding_mask : [bs, hw]         记录Encoder输出特征图的每个位置是否是被pad的
        pos                     : [hw, bs, c]      特征图的位置编码
        query_pos               : [n,  bs, c]      query embedding的位置编码
        """
        # TODO: 实现 Transformer 解码器的前向传播逻辑
        # 1. 遍历$num_layers层TransformerDecoderLayer，对每一层解码器进行前向传播，并处理return_intermediate为True的情况
        ## note: emmmm, intermediate这块的部分逻辑....
        intermediate = []
        x = tgt
        for i, layer in enumerate(self.layers):
            x = layer(x, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
            if self.return_intermediate:
                # intermediate.append(x)
                intermediate.append(self.norm(x))

        # 2. 应用最终的归一化层layer norm
        if self.norm is not None:
            x = self.norm(x)

        # 3. 如果设置了返回中间结果，则将它们堆叠起来返回；否则，返回最终输出
        if self.return_intermediate:
            x = torch.stack(intermediate)
            return x

        return x.unsqueeze(0)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def ffn(self, src):
        layer1 = self.activation(self.linear1(src))
        drop = self.dropout(layer1)
        layer2 = self.linear2(drop)
        return layer2

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        """
        src: [494, bs, 256]  backbone输入下采样32倍后 再压缩维度到256的特征图
        src_mask: None，在Transformer中用来“防作弊”,即遮住当前预测位置之后的位置，忽略这些位置，不计算与其相关的注意力权重
        src_key_padding_mask: [bs, 494]  记录backbone生成的特征图中哪些是原始图像pad的部分 这部分是没有意义的
                           计算注意力会被填充为-inf，这样最终生成注意力经过softmax时输出就趋向于0，相当于忽略不计
        pos: [494, bs, 256]  位置编码
        """
        # TODO: 实现 Transformer 编码器层的前向传播逻辑（参考DETR论文中Section A.3 & Fig.10）
        q = k  = self.with_pos_embed(src, pos)
        v = src

        x, _ = self.self_attn(q, k, v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        x = self.norm1(src + self.dropout1(x))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        """
        src: [494, bs, 256]  backbone输入下采样32倍后 再压缩维度到256的特征图
        src_mask: None，在Transformer中用来“防作弊”,即遮住当前预测位置之后的位置，忽略这些位置，不计算与其相关的注意力权重
        src_key_padding_mask: [bs, 494]  记录backbone生成的特征图中哪些是原始图像pad的部分 这部分是没有意义的
                           计算注意力会被填充为-inf，这样最终生成注意力经过softmax时输出就趋向于0，相当于忽略不计
        pos: [494, bs, 256]  位置编码
        """
        # TODO: 实现 Transformer 编码器层的前向传播逻辑（参考DETR论文中Section A.3 & Fig.10）
        get_src = self.norm1(src)
        k = q = self.with_pos_embed(get_src, pos)
        v = get_src
        x, _ = self.self_attn(q, k, v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        x = self.norm2(src + self.dropout1(x))

        x1 = self.ffn(x)
        x = x + self.dropout2(x1)
        return x

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            # 先对输入进行LN
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def ffn(self, src):
        layer1 = self.activation(self.linear1(src))
        drop = self.dropout(layer1)
        layer2 = self.linear2(drop)
        return layer2

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        """
        tgt: 需要预测的目标 query embedding，负责预测物体
        memory: [h*w, bs, 256]， Encoder的输出，具有全局相关性（增强后）的特征表示
        tgt_mask: None
        memory_mask: None
        tgt_key_padding_mask: None
        memory_key_padding_mask: [bs, h*w]  记录Encoder输出特征图的每个位置是否是被pad的（True无效   False有效）
        pos: [h*w, bs, 256]  encoder输出特征图的位置编码
        query_pos: [100, bs, 256]  query embedding/tgt的位置编码  负责建模物体与物体之间的位置关系  随机初始化的
        tgt_mask、memory_mask、tgt_key_padding_mask是防止作弊的 这里都没有使用
        """
        # TODO: 实现 Transformer 解码器层的前向传播逻辑（参考DETR论文中Section A.3 & Fig.10）
        ## 第一个多头注意力
        q1 = k1 = self.with_pos_embed(tgt, query_pos)
        v1 = tgt
        x1, _ = self.self_attn(q1, k1, v1, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        x1 = self.norm1(tgt + self.dropout1(x1))

        ## 第二个多头注意力
        q2 = self.with_pos_embed(x1, query_pos)
        k2 = self.with_pos_embed(memory, pos)
        v2 = memory
        x2, _ = self.multihead_attn(q2, k2, v2, attn_mask=memory_mask,
                        key_padding_mask=memory_key_padding_mask)
        x3 = self.norm2(x1 + self.dropout2(x2))
        x3 = self.norm3(x3 + self.dropout3(self.ffn(x3)))
        return x3

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        """
        tgt: 需要预测的目标 query embedding，负责预测物体
        memory: [h*w, bs, 256]， Encoder的输出，具有全局相关性（增强后）的特征表示
        tgt_mask: None
        memory_mask: None
        tgt_key_padding_mask: None
        memory_key_padding_mask: [bs, h*w]  记录Encoder输出特征图的每个位置是否是被pad的（True无效   False有效）
        pos: [h*w, bs, 256]  encoder输出特征图的位置编码
        query_pos: [100, bs, 256]  query embedding/tgt的位置编码  负责建模物体与物体之间的位置关系  随机初始化的
        tgt_mask、memory_mask、tgt_key_padding_mask是防止作弊的 这里都没有使用
        """
        # TODO: 实现 Transformer 解码器层的前向传播逻辑（参考DETR论文中Section A.3 & Fig.10）
                ## 第一个多头注意力
        get_tgt = self.norm1(tgt)
        q1 = k1 = self.with_pos_embed(get_tgt, query_pos)
        v1 = get_tgt
        x1, _ = self.self_attn(q1, k1, v1, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(x1)
        x1 = self.norm2(tgt)

        ## 第二个多头注意力
        q2 = self.with_pos_embed(x1, query_pos)
        k2 = self.with_pos_embed(memory, pos)
        v2 = memory
        x2, _ = self.multihead_attn(q2, k2, v2, attn_mask=memory_mask,
                        key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(x2)
        x2 = self.norm3(tgt)
        x = tgt + self.dropout3(self.ffn(x2))
        return x

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            # 先对输入进行LayerNorm
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
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
