import torch


def transfer_parameters(pth_file, finetune_model):
    pretrained_state_dict = torch.load(pth_file)
    # 提取 backbone 部分
    backbone_state_dict = {k: v for k, v in pretrained_state_dict.items() if 'head' not in k}
    finetune_model.backbone.load_state_dict(backbone_state_dict, strict=False)
    for param in finetune_model.backbone.parameters():
        param.requires_grad = False
    return finetune_model