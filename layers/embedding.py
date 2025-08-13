import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_size, seq_len=1000):
        super(PositionalEmbedding, self).__init__()
        # 位置序列
        position = torch.arange(seq_len).float().unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size))
        # 单双数处理，且位置编码不学习，保证通用性
        pe = torch.zeros((seq_len, embed_size), dtype=torch.float32)
        pe.require_grad = False
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 便于批处理
        pe = pe.unsqueeze(0)
        # 注册为一个缓冲区，这样它不会被视为模型的参数，但仍然会在模型保存和加载时被包含
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class CrossEmbedding(nn.Module):
    def __init__(self, seq_len, embed_size):
        super(CrossEmbedding, self).__init__()
        self.seq_len = seq_len
        self.embed_size = embed_size
        self.var_emb = nn.Linear(seq_len, embed_size)

    def forward(self, x):
        """
        cross变量嵌入
        :param x: [batchsize, n_feature, seq_len]
        :return: [batchsize, n_feature, embed_size]
        """
        x = self.var_emb(x)
        return x

if __name__ == "__main__":
    seq_len = 100
    embed_size = 16
    batchsize = 4
    n_feature = 21
    patch_len = 5

    pos_emb = PositionalEmbedding(seq_len, embed_size)
    x = torch.randn(batchsize, n_feature, seq_len)
    x = x.unfold(dimension=-1, size=patch_len, step=patch_len)
    print(x.shape)
    x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
    emb = pos_emb(x)
    print(emb.shape)