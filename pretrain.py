from data_process import TimeSeriesDataset
from model.PatchTST import PatchTST
from model.GlobalPatchTST import GlobalPatchTST
from utils.metric import masked_mse_loss, masked_mae_loss

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
import swanlab

n_feature_map = {
    "weather": 21,
    "traffic": 862,
    "electricity": 321
}

def model_select(name):
    if name == "PatchTST":
        pretrain_model = PatchTST
    else:
        pretrain_model = GlobalPatchTST
    return pretrain_model

def main(args):
    # 设备设置
    n_feature = n_feature_map[args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 导入数据
    csv_path = f"./data/{args.dataset}/{args.dataset}.csv"
    dataset = TimeSeriesDataset(csv_path, n_feature, args.seq_len, args.pred_len)
    # 可复现数据集
    torch.manual_seed(42)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    # 预训练模型
    pretrain_model = model_select(args.model_name)
    model = pretrain_model(
        seq_len=args.seq_len,
        n_feature=n_feature,
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
    # 设置模型参数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = masked_mse_loss
    mae_recorder = masked_mae_loss
    model.train()
    # swanlab记录实验
    if args.swanlab:
        swanlab.init(
            project="PatchTST",
            name=f"PatchTST_{args.dataset}_pretrain",
            config=args,
        )
    # pretrain
    for epoch in range(args.epochs):
        for step, (lookback, pred) in enumerate(data_loader):
            lookback, pred = lookback.to(device), pred.to(device)
            optimizer.zero_grad()
            origin, output, mask, _ = model(lookback)
            loss = criterion(origin, output, mask, device=device)
            mae_loss = mae_recorder(origin, output, mask, device=device)
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print(f"Epoch [{epoch + 1}/{args.epochs}], Step: {step}, Loss: {loss.item(): .4f}")
            if args.swanlab:
                swanlab.log({"train/MSE": loss.item(), "train/MAE": mae_loss.item()})

        torch.save(model.state_dict(), f"{args.save_dir}/TST_on_{args.dataset}_{epoch+1}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PatchTST model")
    parser.add_argument("--dataset", type=str, choices=["weather", "traffic", "electricity"],
                        default="weather", help="Dataset name")
    parser.add_argument("--model_name", type=str, choices=["PatchTST", "GlobalPatchTST"], default="GlobalPatchTST",
                        help="Model name, options: PatchTST, GlobalPatchTST")
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
    parser.add_argument("--mask_ratio", type=float, default=0.4, help="Mask ratio for pre-training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--padding", action="store_true", help="Use padding in patching")
    parser.add_argument("--swanlab", action="store_true",
                        help="Use SWANLab for training")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    args = parser.parse_args()
    main(args)
