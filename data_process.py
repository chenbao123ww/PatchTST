from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path

data_dir = Path(__file__).parent / "data"

dataset_list = [
    "weather",
    "traffic",
    "electricity"
]

class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path, n_feature, seq_len=336, pred_len=96):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_feature = n_feature

        data = pd.read_csv(csv_path).values
        # print(data.shape)
        start_col = data.shape[1] - n_feature
        data = data[:, start_col:].astype(np.float32)
        self.data = torch.tensor(data, dtype=torch.float32)

        # 确认形状
        assert self.data.shape[1] == n_feature, "Number of features in data does not match n_feature"

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        lookback_sample = self.data[idx:idx + self.seq_len].T
        pred_sample = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len].T
        return lookback_sample, pred_sample

if __name__ == "__main__":
    dataset_name = dataset_list[0]
    csv_path = data_dir / dataset_name / (dataset_name + ".csv")
    pred_len = 24
    seq_len = 50
    n_feature = 21

    dataset = TimeSeriesDataset(csv_path, n_feature, seq_len, pred_len)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in dataloader:
        lookback, pred = batch  # 解包
        # print("Lookback shape:", lookback.shape)
        # print("Prediction shape:", pred.shape)