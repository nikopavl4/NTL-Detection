from typing import Union, Optional

import torch


class FocalLoss(torch.nn.Module):
    """
    Focal loss for imbalanced classification.

    Args:
        alpha (float, optional): Scaling factor for the rare class. Default to 1.
        gamma (float, optional): Focusing parameter. Defaults to 2.
        reduction (Union[str, None], optional): The reduction to apply to the output. Defaults to mean.
    """

    def __init__(self, alpha: Optional[float] = 1., gamma: Optional[float] = 2.,
                 reduction: Optional[Union[str, None]] = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # get the bce loss with logits without reduction
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        # get the probability term from the BCE loss
        pt = torch.exp(-bce_loss)
        # compute the focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        # apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CustomBCELoss(torch.nn.Module):
    """
    Custom binary cross entropy loss with increased penalty for false negatives.

    Args:
        false_negative_weight (float): Weight to penalize false negatives more.
        reduction (Union[str, None], optional): The reduction to apply to the output. Defaults to 'mean'.
    """

    def __init__(self, false_negative_weight: float, reduction: Optional[Union[str, None]] = 'mean'):
        super(CustomBCELoss, self).__init__()
        self.false_negative_weight = false_negative_weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # calculate the standard BCE loss without reduction
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # create a weight tensor that has the false_negative_weight for class 1 (false negatives) and 1 for class 0
        weights = torch.where(targets == 1, self.false_negative_weight * torch.ones_like(targets),
                              torch.ones_like(targets))

        # apply the weights to the BCE loss
        weighted_bce_loss = weights * bce_loss

        # apply reduction
        if self.reduction == 'mean':
            return weighted_bce_loss.mean()
        elif self.reduction == 'sum':
            return weighted_bce_loss.sum()
        else:
            return weighted_bce_loss


class CustomHingeLoss(torch.nn.Module):
    """
    Custom hinge loss for binary classification with increased penalty for false negatives.

    Args:
        false_negative_weight (float): Weight to penalize false negatives more.
        margin (float, optional): Margin for the hinge loss. Defaults to 1.
        reduction (Union[str, None], optional): The reduction to apply to the output. Defaults to 'mean'.
    """

    def __init__(self, false_negative_weight: float, margin: Optional[float] = 1.,
                 reduction: Optional[Union[str, None]] = 'mean'):
        super(CustomHingeLoss, self).__init__()
        self.false_negative_weight = false_negative_weight
        self.margin = margin
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # convert targets from [0, 1] to [-1, 1]
        targets = 2 * targets - 1

        # compute hinge loss
        hinge_loss = torch.clamp(1 - inputs * targets, min=0)

        # create a weight tensor that has the false_negative_weight for class 1 (false negatives) and 1 for class 0
        weights = torch.where(targets == 1, self.false_negative_weight * torch.ones_like(targets),
                              torch.ones_like(targets))

        # apply the weights to the hinge loss
        weighted_hinge_loss = weights * hinge_loss

        # apply reduction
        if self.reduction == 'mean':
            return weighted_hinge_loss.mean()
        elif self.reduction == 'sum':
            return weighted_hinge_loss.sum()
        else:
            return weighted_hinge_loss


class ModifiedHuberLoss(torch.nn.Module):
    """
    Modified Huber loss for binary classification.

    Args:
        reduction (Union[str, None], optional): The reduction to apply to the output. Defaults to 'mean'.
    """

    def __init__(self, reduction: Optional[Union[str, None]] = 'mean'):
        super(ModifiedHuberLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # convert targets from [0, 1] to [-1, 1]
        targets = 2 * targets - 1

        # compute modified huber loss
        loss = torch.where(targets * inputs < -1,
                           -4 * targets * inputs,
                           (1 - targets * inputs) ** 2)

        # apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
