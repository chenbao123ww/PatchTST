import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, n_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_features))
            self.bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x, mode):
        if mode == "norm":
            dim2reduce = tuple(range(1, x.ndim - 1))
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            x = x - self.mean
            x = x / self.stdev
            if self.affine:
                x = x * self.weight
                x = x + self.bias
        elif mode == "denorm":
            if self.affine:
                x = x - self.bias
                x = x / (self.weight + self.eps * self.eps)
            x = x * self.stdev
            x = x + self.mean

        return x


if __name__ == '__main__':
    N, L, C = 8, 100, 64            # [batchsize, seq_len, n_feature]
    input_tensor = torch.randn(N, L, C)
    model = RevIN(n_features=C)
    # print(input_tensor[0, :, 0])

    normalized_output = model(input_tensor, mode="norm")
    # print("Normalized Output Shape:", normalized_output.shape)
    # print(normalized_output[0, :, 0])

    original_output = model(normalized_output, mode="denorm")
    # print("Original Output Shape:", original_output.shape)
    # print(original_output[0, :, 0])
