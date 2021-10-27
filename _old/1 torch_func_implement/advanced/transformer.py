# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch.nn as nn
from torch.nn import Parameter
import torch
from torch import Tensor
import torch.nn.functional as F
import math


# 此处参考:
# 1. torch.nn.Transformer
# 2. https://arxiv.org/abs/1706.03762
# 3. http://nlp.seas.harvard.edu/2018/04/03/attention.html
# 为了避免与torch.nn中的函数或类混淆，自己复现的函数或类前加上`_`做区分
class _MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super(_MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        # project: 映射
        self.in_proj_list = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim, True),  # query_proj
            nn.Linear(embed_dim, embed_dim, True),  # key_proj
            nn.Linear(embed_dim, embed_dim, True)  # value_proj
        ])
        self.out_proj = nn.Linear(embed_dim, embed_dim, True)

    def forward(self, query, key, value,
                key_padding_mask=None,
                need_weights=True, attn_mask=None):
        """

        :param query: shape[TL, N, E]
        :param key: shape[SL, N, E]
        :param value: shape[SL, N, E]
        :param key_padding_mask: shape[N, SL]
        :param need_weights: bool
        :param attn_mask: shape[TL, SL]
        :return: Tuple[output: shape[TL, N, E], output_weight: shape[N, TL, SL]]
        """
        num_heads = self.num_heads
        dropout_p = self.dropout_p
        training = self.training
        tgt_len, batch_size, embed_dim = query.shape
        src_len = key.shape[0]
        head_dim = embed_dim // num_heads  # 需要可以被整除, 此处不进行检查
        # shape[TL, N, E], shape[SL, N, E], shape[SL, N, E]
        query, key, value = self.in_proj_list[0](query), self.in_proj_list[1](key), self.in_proj_list[2](value)
        # shape[N * NH, TL, HD], shape[N * NH, SL, HD], shape[N * NH, SL, HD]
        query = query.contiguous().view(tgt_len, batch_size * num_heads, head_dim).transpose(0, 1)
        key = key.contiguous().view(src_len, batch_size * num_heads, head_dim).transpose(0, 1)
        value = value.contiguous().view(src_len, batch_size * num_heads, head_dim).transpose(0, 1)
        scale = 1 / math.sqrt(head_dim)
        query = query * scale
        # shape[N * NH, TL, SL]. the weights on the values
        attn_output_weights = query @ key.transpose(1, 2)
        if attn_mask is not None:  # TL, SL位置上. decoder层面. [TL, SL] or [N * NH, TL, SL]
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        if key_padding_mask is not None:  # key上. 任务层面
            # shape[N, NH, TL, SL]
            attn_output_weights = attn_output_weights.view(batch_size, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask[:, None, None, :],  # [N, 1, 1, SL]
                float("-inf"),
            )
            # shape[N * NH, TL, SL]
            attn_output_weights = attn_output_weights.view(batch_size * num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, dropout_p, training)
        # shape[N * NH, TL, HD].
        attn_output = attn_output_weights @ value  # [N * NH, TL, SL] @ [N * NH, SL, HD]
        # shape[TL, N, E]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, batch_size, embed_dim)
        attn_output = self.out_proj(attn_output)  # 此处已Concat. 直接全连接
        if need_weights:
            # [N, NH, TL, SL]
            attn_output_weights = attn_output_weights.view(batch_size, num_heads, tgt_len, src_len)
            # [N, TL, SL]
            attn_output_weights = torch.mean(attn_output_weights, 1)
            return attn_output, attn_output_weights
        else:
            return attn_output, None


class _TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout_p=0.1):
        super(_TransformerEncoderLayer, self).__init__()
        # sub_layer1
        self.self_attn = _MultiheadAttention(d_model, num_heads, dropout_p)
        self.dropout1 = nn.Dropout(dropout_p)
        self.norm1 = nn.LayerNorm(d_model)
        # sub_layer2
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout_p)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src: Tensor, src_mask=None, src_key_padding_mask=None):
        """

        :param src: shape[SL, N, E]
        :param src_mask: shape[SL, SL]
        :param src_key_padding_mask: shape[N, SL]
        :return: shape[SL, N, E]
        """
        # sub_layer1
        src0 = src
        src = self.self_attn(src, src, src, src_key_padding_mask, False, src_mask)[0]
        src = src0 + self.dropout1(src)
        src = self.norm1(src)
        # sub_layer2
        src0 = src
        src = self.ffn(src)
        src = src0 + self.dropout2(src)
        src = self.norm2(src)
        return src


class _TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout_p=0.1):
        super(_TransformerDecoderLayer, self).__init__()
        # sub_layer1
        self.self_attn = _MultiheadAttention(d_model, num_heads, dropout_p=dropout_p)
        self.dropout1 = nn.Dropout(dropout_p)
        self.norm1 = nn.LayerNorm(d_model)
        # sub_layer2
        self.multihead_attn = _MultiheadAttention(d_model, num_heads, dropout_p=dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        self.norm2 = nn.LayerNorm(d_model)
        # sub_layer3
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout3 = nn.Dropout(dropout_p)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """

        :param tgt: shape[TL, N, E]. embedding + positional encoding
        :param memory: shape[SL, N, E]. encoder的输出
        :param tgt_mask: shape[TL, TL]
        :param memory_mask: shape[TL, SL]
        :param tgt_key_padding_mask: shape[N, TL]
        :param memory_key_padding_mask: shape[N, SL]
        :return: shape[TL, N, E]. 未过linear 和 softmax
        """
        # sub_layer1
        tgt0 = tgt
        tgt = self.self_attn(tgt, tgt, tgt, tgt_key_padding_mask, False, tgt_mask)[0]
        tgt = tgt0 + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        # sub_layer2
        tgt0 = tgt
        tgt = self.multihead_attn(tgt, memory, memory, memory_key_padding_mask, False, memory_mask)[0]
        tgt = tgt0 + self.dropout2(tgt)
        tgt = self.norm2(tgt)
        # sub_layer3
        tgt0 = tgt
        tgt = self.ffn(tgt)
        tgt = tgt0 + self.dropout3(tgt)
        tgt = self.norm3(tgt)
        return tgt


class _TransformerBackbone(nn.Module):
    """未加入embedding与positional encoding, 以及最后的Linear和softmax.
    同torch.nn.Transformer"""

    def __init__(self, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout_p=0.1):
        super(_TransformerBackbone, self).__init__()
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.encoder_list = nn.ModuleList([
            _TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout_p)
            for _ in range(num_encoder_layers)
        ])
        self.norm1 = nn.LayerNorm(d_model)
        self.decoder_list = nn.ModuleList([
            _TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout_p)
            for _ in range(num_decoder_layers)
        ])
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, tgt,
                src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """

        :param src: shape[SL, N, E]
        :param tgt: shape[TL, N, E]
        :param src_mask: shape[SL, SL]
        :param tgt_mask: shape[TL, TL]
        :param memory_mask: shape[TL, SL]
        :param src_key_padding_mask: shape[N, SL]
        :param tgt_key_padding_mask: shape[N, TL]
        :param memory_key_padding_mask: shape[N, SL]
        :return: shape[TL, N, E]. 未过linear 和 softmax
        """
        # encoder
        for i in range(self.num_encoder_layers):
            src = self.encoder_list[i](src, src_mask, src_key_padding_mask)
        memory = self.norm1(src)
        del src
        # decoder
        for i in range(self.num_encoder_layers):
            tgt = self.decoder_list[i](tgt, memory, tgt_mask, memory_mask,
                                       tgt_key_padding_mask, memory_key_padding_mask)
        tgt = self.norm2(tgt)
        return tgt


class _EmbeddingWithPositionalEncoding(nn.Module):
    def __init__(self, num_vocabs, max_len=500, d_model=512, dropout=0.1):
        """

        :param num_vocabs: 词汇表长度
        :param max_len: 句子最大长度
        :param d_model: E
        :param dropout:
        """
        super(_EmbeddingWithPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_vocabs, d_model)
        self.pe = self.generate_pe(max_len, d_model)  # positional encoding. shape[ML, 1, E]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """

        :param x: shape[SL, N]. Long
        :return: shape[SL, N, E]. Float32
        """
        x = self.embedding(x) * math.sqrt(self.d_model)  # shape[SL, N, E]
        pe = self.pe[:x.shape[0]]  # shape[SL, N, E]
        x = x + pe  # shape[SL, N, E]
        x = self.dropout(x)
        return x

    @staticmethod
    def generate_pe(max_len, d_model=512):
        pe = torch.zeros(max_len, 1, d_model)  # [SL, N, E]
        position = torch.arange(0, max_len)  # [MAX_LEN]
        div_term = 1 / 10000 ** (torch.arange(0, d_model, 2) / d_model)  # [E/2]
        pe[:, 0, 0::2] = torch.sin(position * div_term[None])  # [MAX_LEN, E/2]
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe


# _EmbeddingWithPositionalEncoding(30000)(torch.randint(0, 30000, (32, 1)))

if __name__ == "__main__":
    torch.manual_seed(0)
    m0 = nn.Transformer()
    m1 = _TransformerBackbone()
    m0_name_list = []
    m1_name_list = []
    m0_state_dict = m0.state_dict()
    m1_state_dict = m1.state_dict()
    with open("layers_map/1.txt", "r") as f:
        for line in f:
            m0_name_list.append(line[:-1])
    with open("layers_map/2.txt", "r") as f:
        for line in f:
            m1_name_list.append(line[:-1])
    for (n0, n1) in zip(m0_name_list, m1_name_list):
        prefix_n1, suffix_n1 = n1.rsplit('.', maxsplit=1)
        if suffix_n1 == "in_proj_weight":
            m1_state_dict[prefix_n1 + '.in_proj_list.0.weight'], \
            m1_state_dict[prefix_n1 + '.in_proj_list.1.weight'], \
            m1_state_dict[prefix_n1 + '.in_proj_list.2.weight'] = torch.chunk(m0_state_dict[n0], 3, 0)
        elif suffix_n1 == "in_proj_bias":
            m1_state_dict[prefix_n1 + '.in_proj_list.0.bias'], \
            m1_state_dict[prefix_n1 + '.in_proj_list.1.bias'], \
            m1_state_dict[prefix_n1 + '.in_proj_list.2.bias'] = torch.chunk(m0_state_dict[n0], 3, 0)
        else:
            m1_state_dict[n1] = m0_state_dict[n0]
    m1.load_state_dict(m1_state_dict)
    torch.manual_seed(0)
    src = torch.rand(10, 16, 512)
    tgt = torch.rand(20, 16, 512)
    src_mask = (torch.rand(10, 10) * 1.1).floor().bool()
    tgt_mask = (torch.rand(20, 20) * 1.1).floor().bool()
    memory_mask = (torch.rand(20, 10) * 1.1).floor().bool()
    src_key_padding_mask = (torch.rand(16, 10) * 1.1).floor().bool()
    tgt_key_padding_mask = (torch.rand(16, 20) * 1.1).floor().bool()
    memory_key_padding_mask = (torch.rand(16, 10) * 1.1).floor().bool()
    torch.manual_seed(0)
    y0 = m0(src, tgt, src_mask, tgt_mask, memory_mask,
            src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
    torch.manual_seed(0)
    y1 = m1(src, tgt, src_mask, tgt_mask, memory_mask,
            src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
    print(m1)
    # print(y0)
    # print(y1)
    print(torch.all(torch.abs(y1 - y0) < 1e-4))
