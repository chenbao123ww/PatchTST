from layers.module import MultiHeadAttention, FeedForward
import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, hidden_size=768, bias=True, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_size, heads, bias=bias)
        self.feed_forward = FeedForward(embed_size, hidden_size, bias=bias, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x, mask=None):
        attention = self.self_attn(x, x, x, mask)
        x = self.norm1(attention + x)
        forward = self.feed_forward(x)
        x = self.norm2(forward + x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, hidden_size, heads, num_layers, bias=True, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(embed_size, heads, hidden_size, bias=bias, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
