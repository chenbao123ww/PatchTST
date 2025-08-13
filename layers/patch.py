import random

import torch
import torch.nn as nn


class Patcher():
    def __init__(self, seq_len, patch_len, stride, padding=False):
        self.patch_len = patch_len
        self.stride = stride
        self.num_patch = int((seq_len - patch_len)/stride + 1) + (1 if padding else 0)
        self.padding = padding
        if self.padding:
            self.padding_layer = nn.ReplicationPad1d((0, stride))

    def create(self, x):
        """
        x: [batchsize, n_feature, seq_len]
        return: [batchsize, n_feature, num_patch, patch_len]
        """
        if self.padding:
            self.padding_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return x

    def mask(self, x, mask_ratio=0.4):
        """
        x: [batchsize, n_feature, num_patch, patch_len]
        """
        batch_size, n_feature, num_patch, patch_len = x.shape
        num_masked = int(num_patch * mask_ratio)

        mask = torch.zeros((batch_size, n_feature, num_patch), dtype=torch.bool)
        for i in range(batch_size):
            for j in range(n_feature):
                mask_indices = torch.randperm(num_patch)[:num_masked]
                mask[i, j, mask_indices] = True

        masked_x = x.clone()
        masked_x[mask] = 0.0

        return masked_x, mask


if __name__ == "__main__":
    random.seed(42)
    patch_len = 16
    stride = 8
    batchsize = 32
    seq_len = 36
    n_feature = 10
    patcher = Patcher(seq_len, patch_len, stride, padding=True)
    time_series = torch.rand((batchsize, n_feature, seq_len))
    patches = patcher.create(time_series)
    masked_tensor, mask = patcher.mask(patches)
    # print(patches)
    # print(patches.shape)
    print(masked_tensor)
    print(mask.shape)
