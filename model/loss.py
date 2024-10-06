import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def bce_withlogits_loss(output, target):
    r"""L2正则化的二元交叉熵损失函数, 防止过拟合"""
    return F.binary_cross_entropy_with_logits(output, target)