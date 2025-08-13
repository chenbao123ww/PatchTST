import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads, bias=True, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        # 要求embed_size能被heads整除
        assert self.head_dim * heads == embed_size, "Embedding size needs to be divisible by heads"
        # q,k,v投影
        self.queries = nn.Linear(embed_size, embed_size, bias=bias)
        self.keys = nn.Linear(embed_size, embed_size, bias=bias)
        self.values = nn.Linear(embed_size, embed_size, bias=bias)
        self.fc_out = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, query, key, value, mask=None):
        """
        :param q,k,v: [batchsize, seq_length, embed_size]
        :param mask: [batchsize, seq_length, seq_length]
        """
        # 自注意力下qkv长度相等，交叉注意力下q和kv长度不等但是kv长度相等
        batchsize, q_length, _ = query.shape
        _, k_length, _ = key.shape
        v_length = k_length
        queries = self.queries(query)     # [batchsize, seq_length, embed_size]
        keys = self.keys(key)
        values = self.values(value)
        # 拆分到多头上，[batchsize, seq_length, head, head_dim]
        queries = queries.view(batchsize, q_length, self.heads, self.head_dim)
        keys = keys.view(batchsize, k_length, self.heads, self.head_dim)
        values = values.view(batchsize, v_length, self.heads, self.head_dim)
        # 变形用于后续做点积，[batchsize, head, seq_length, head_dim]
        queries = queries.permute(0, 2, 1, 3)  # [batchsize, head, seq_length, head_dim]
        keys = keys.permute(0, 2, 3, 1)         # [batchsize, head, head_dim, seq_length]
        values = values.permute(0, 2, 1, 3)

        # 输出注意力[batchsize, head, q_length, k_length]
        energy = torch.matmul(queries, keys)
        if mask is not None:
            mask = mask.unsqueeze(1)
            energy = energy.masked_fill(mask == 0, float('-inf'))
        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=-1)
        # 计算出注意力，和v点积，[batchsize, head, q_length, head_dim]
        out = torch.matmul(attention, values)
        # 各头组合，[batchsize, q_length, embed_size]
        out = out.transpose(1, 2).contiguous().view(batchsize, q_length, self.embed_size)
        out = self.fc_out(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden_size, bias=True, dropout=0.1):
        super(FeedForward, self).__init__()
        # 稠密层作为前馈网络
        self.ff = nn.Sequential(
            nn.Linear(embed_size, hidden_size, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_size, bias=bias),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)
