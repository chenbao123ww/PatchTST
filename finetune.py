from data_process import TimeSeriesDataset
from model.PatchTST import PatchTST, PatchTSTPred
from utils.metric import masked_mse_loss, masked_mae_loss
from utils.param import transfer_parameters

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
import swanlab


def main(args):
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 导入数据
    csv_path = f"./data/{args.dataset}/{args.dataset}.csv"
    dataset = TimeSeriesDataset(csv_path, args.n_feature, args.seq_len, args.pred_len)
    # 可复现数据集
    torch.manual_seed(42)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    # 预训练模型
    model = PatchTST(
        seq_len=args.seq_len,
        n_feature=args.n_feature,
        patch_len=args.patch_len,
        stride=args.stride,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout,
        affine=args.affine,
        mask_ratio=args.mask_ratio,
        padding=args.padding,
    ).to(device)
    # 下游预测模型
    pred_model = PatchTSTPred(
        backbone=model,
        pred_len=args.pred_len,
    ).to(device)

    pred_model = transfer_parameters(args.pretrain_model, pred_model)
    # swanlab记录实验
    if args.swanlab:
        swanlab.init(
            project="PatchTST",
            name=f"PatchTST_{args.dataset}_finetune",
            config=args,
        )

    # 设置模型参数
    optimizer = optim.Adam(pred_model.parameters(), lr=args.lr)
    criterion = masked_mse_loss
    mae_recorder = masked_mae_loss
    pred_model.train()
    # finetune
    for epoch in range(args.epochs):
        for step, (lookback, target) in enumerate(data_loader):
            lookback, target = lookback.to(device), target.to(device)
            optimizer.zero_grad()
            pred = pred_model(lookback)
            loss = criterion(pred, target)
            mae_loss = mae_recorder(pred, target)
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print(f"Epoch: {epoch+1}, Test Step: {step}, Loss: {loss.item(): .4f}")
            if args.swanlab:
                swanlab.log({"test/MSE": loss.item(), "test/MAE": mae_loss.item()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune PatchTST model")
    parser.add_argument("--dataset", type=str, default="weather", help="Dataset name")
    parser.add_argument("--n_feature", type=int, default=21,
                        help="Number of features in the dataset, weather=21, traffic=862, electricity=321")
    parser.add_argument("--seq_len", type=int, default=512, help="Length of the input sequence")
    parser.add_argument("--pred_len", type=int, default=96, help="Length of the prediction sequence")
    parser.add_argument("--patch_len", type=int, default=12, help="Length of each patch")
    parser.add_argument("--stride", type=int, default=12, help="Stride for patching")
    parser.add_argument("--embed_size", type=int, default=128, help="Embedding size")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size in Transformer")
    parser.add_argument("--affine", action="store_true", help="Use affine transformation in RevIN")
    parser.add_argument("--n_layer", type=int, default=3, help="Number of Transformer layers")
    parser.add_argument("--n_head", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--mask_ratio", type=float, default=0, help="Mask ratio for pre-training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for fine-tuning")
    parser.add_argument("--padding", action="store_true", help="Use padding in patching")
    parser.add_argument("--swanlab", action="store_true",
                        help="Use SWANLab for training")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--pretrain_model", type=str, default=None,
                        help="Path to pre-trained model weights")
    args = parser.parse_args()
    main(args)
