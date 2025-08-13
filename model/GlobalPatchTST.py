import torch
import torch.nn as nn
from layers.module import MultiHeadAttention, FeedForward
from layers.embedding import PositionalEmbedding, CrossEmbedding
from utils.revin import RevIN
from utils.patch import Patcher


class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, hidden_size, bias=True, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_size, heads, bias=bias)
        self.cross_attn = MultiHeadAttention(embed_size, heads, bias=bias)
        self.feed_forward = FeedForward(embed_size, hidden_size, bias=bias, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        """
        :param x: [batchsize * n_feature, num_patch+1, embed_size]
        :param cross: [batchsize, n_feature, embed_size]
        :return:
        """
        B, L, D = cross.shape
        x = x + self.dropout(self.self_attn(x, x, x, x_mask))
        x = self.norm1(x)
        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attn(x_glb, cross, cross, cross_mask))
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = self.feed_forward(y)
        y = self.norm3(x + y)

        return y


class GlobalEncoder(nn.Module):
    def __init__(self, embed_size, heads, hidden_size, n_layer, bias=True, dropout=0.1):
        super(GlobalEncoder, self).__init__()
        self.encoder = nn.ModuleList([
            EncoderLayer(embed_size, heads, hidden_size, bias=bias, dropout=dropout)
            for _ in range(n_layer)
        ])

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.encoder:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        return x


class GlobalPatchTSTEncoder(nn.Module):
    def __init__(self, patch_len, num_patch, n_feature,
                 embed_size=128, hidden_size=256,
                 n_layer=3, n_head=16, dropout=0., bias=True):
        super(GlobalPatchTSTEncoder, self).__init__()
        self.lin_pro = nn.Linear(patch_len, embed_size)
        self.pos_emb = PositionalEmbedding(embed_size)
        self.global_emb = nn.Parameter(torch.randn(1, n_feature, 1, embed_size))
        self.encoder = GlobalEncoder(embed_size, n_head, hidden_size, n_layer,
                                     bias=bias, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross):
        """
        x: [batchsize, n_feature, num_patch, patch_len] from patch.py
        return: [batchsize, n_feature, embed_size, num_patch]
        """
        glb = self.global_emb.repeat((x.shape[0], 1, 1, 1))         # [batchsize, n_feature, 1, embed_size]
        batchsize, n_feature, num_patch, patch_len = x.shape

        x = torch.reshape(x, (batchsize * n_feature, num_patch, patch_len))
        x = self.lin_pro(x) + self.pos_emb(x)                       # [batchsize * n_feature, num_patch, embed_size]
        x = torch.reshape(x, (batchsize, n_feature, num_patch, -1))  # [batchsize, n_feature, num_patch, embed_size]
        x = torch.cat([x, glb], dim=2)                              # [batchsize, n_feature, num_patch+1, embed_size]
        x = torch.reshape(x, (batchsize * n_feature, num_patch+1, -1))  # [batchsize * n_feature, num_patch+1, embed_size]
        x = self.encoder(x, cross)                                         # [batchsize, n_feature, num_patch+1, embed_size]
        x = torch.reshape(x, (batchsize, n_feature, num_patch+1, -1))     # [batchsize, n_feature, num_patch+1, embed_size]
        x = self.dropout(x)
        return x

class GlobalPatchTST(nn.Module):
    def __init__(self, seq_len, n_feature, patch_len, stride,
                 embed_size, hidden_size, n_layer=3, n_head=16,
                 dropout=0., mask_ratio=0.4, eps=1e-5,
                 bias=True, padding=True, affine=True, revin=True):
        super(GlobalPatchTST, self).__init__()
        self.seq_len = seq_len
        self.n_feature = n_feature
        self.patch_len = patch_len
        self.stride = stride
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.revin = revin
        self.mask_ratio = mask_ratio
        self.dropout = dropout
        if self.revin:
            self.revin_layer = RevIN(n_feature, eps=eps, affine=affine)
        self.patcher = Patcher(seq_len, patch_len, stride, padding=padding)
        self.num_patch = self.patcher.num_patch
        self.cross_emb = CrossEmbedding(seq_len, embed_size)
        self.patch_encoder = GlobalPatchTSTEncoder(patch_len, self.num_patch, n_feature, embed_size, hidden_size,
                                                    n_layer=n_layer, n_head=n_head, dropout=dropout, bias=bias)

        self.pro_head = PretrainHead(embed_size, hidden_size, patch_len, dropout=dropout)

    def forward(self, x):
        """
        x: [batchsize, n_feature, seq_len]
        return:
        x: [batchsize, n_feature, num_patch, patch_len] to mask for self-supervised
        u: [batchsize, n_feature, num_patch, patch_len] result for construction
        """
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, mode="norm")
            x = x.permute(0, 2, 1)
        cross = self.cross_emb(x)
        x_patch = self.patcher.create(x)      # [batchsize, n_feature, num_patch, patch_len]
        masked_x, mask = self.patcher.mask(x_patch, mask_ratio=self.mask_ratio)
        encoded_x = self.patch_encoder(masked_x, cross)[:, :, :self.num_patch, :]   # [batchsize, n_feature, num_patch, embed_size]
        u = self.pro_head(encoded_x)                               # [batchsize, n_feature, num_patch, patch_len]

        return x_patch, u, mask, encoded_x


class GlobalPatchTSTPred(nn.Module):
    def __init__(self, backbone: GlobalPatchTST, pred_len):
        super().__init__()
        self.backbone = backbone
        num_patch = self.backbone.patcher.num_patch
        self.revin = self.backbone.revin
        n_feature = self.backbone.n_feature
        embed_size = self.backbone.embed_size
        dropout = self.backbone.dropout
        self.pred_head = PredictionHead(n_feature, embed_size, num_patch, pred_len, dropout=dropout)

    def forward(self, x):
        _, _, _, encoded_x = self.backbone(x)
        out = self.pred_head(encoded_x)           # supervised: [batchsize, n_feature, pred_len]
        if self.revin:
            out = out.permute(0, 2, 1)              # [batchsize, pred_len, n_feature]
            out = self.backbone.revin_layer(out, mode="denorm")
            out = out.permute(0, 2, 1)              # [batchsize, n_feature, pred_len]

        return out

class PretrainHead(nn.Module):
    def __init__(self, embed_size, hidden_size, patch_len, dropout=0.):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_size, hidden_size),  # 第一个全连接层，128 是隐藏节点数
            nn.ReLU(),                   # 激活函数
            nn.Linear(hidden_size, patch_len)   # 输出层，输出维度为 output_dim
        )

    def forward(self, x):       # [batchsize, n_feature, num_patch, embed_size]
        x = self.layer(x)       # [batchsize, n_feature, num_patch, patch_len]
        return x


class PredictionHead(nn.Module):
    def __init__(self, n_feature, embed_size, num_patch, pred_len, dropout=0.):
        super().__init__()
        self.n_feature = n_feature
        self.linears = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.flattens = nn.ModuleList()
        for i in range(self.n_feature):
            self.flattens.append(nn.Flatten(start_dim=-2))
            self.linears.append(nn.Linear(embed_size * num_patch, pred_len))
            self.dropouts.append(nn.Dropout(dropout))

    def forward(self, x):           # [batchsize, n_feature, num_patch, embed_size]
        x = x.permute(0, 1, 3, 2)   # [batchsize, n_feature, embed_size, num_patch]
        x_out = []
        for i in range(self.n_feature):
            z = self.flattens[i](x[:, i, :, :])
            z = self.linears[i](z)
            z = self.dropouts[i](z)
            x_out.append(z)
        x = torch.stack(x_out, dim=1)
        return x
