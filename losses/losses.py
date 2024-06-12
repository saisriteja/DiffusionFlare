import torch


def L1_loss(pred, target):
    return torch.mean(torch.abs(pred - target))
