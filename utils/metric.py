import torch
import torch.nn.functional as F

def masked_mse_loss(pred, target, mask=None):
    assert pred.shape == target.shape, "Prediction and target must have the same shape."

    loss = F.mse_loss(pred, target, reduction='none')

    mask_num = mask.sum() if mask is not None else 0

    if mask is not None:
        mask = mask.unsqueeze(-1)       # [batch_size, n_feature, num_patch, 1]
        loss = loss * mask.float()

    loss = loss.sum() / mask.sum() if mask_num > 0 else loss.mean()

    return loss


def masked_mae_loss(pred, target, mask=None):
    assert pred.shape == target.shape, "Prediction and target must have the same shape."

    loss = torch.abs(pred - target)

    mask_num = mask.sum() if mask is not None else 0

    if mask is not None:
        mask = mask.unsqueeze(-1)       # [batch_size, n_feature, num_patch, 1]
        loss = loss * mask.float()

    loss = loss.sum() / mask.sum() if mask_num > 0 else loss.mean()

    return loss
