import torch
import torch.nn as nn
import torch.nn.functional as F


def ssim_loss(predictions, targets, C1=0.01**2, C2=0.03**2):
    """
    SSIM loss adaptada para imágenes RGB en formato:
    [batch, channels, height, width]

    Las imágenes deben estar normalizadas en rango [0, 1].
    """

    mu_x = F.avg_pool2d(predictions, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(targets, kernel_size=3, stride=1, padding=1)

    sigma_x = (
        F.avg_pool2d(predictions**2, kernel_size=3, stride=1, padding=1)
        - mu_x**2
    )

    sigma_y = (
        F.avg_pool2d(targets**2, kernel_size=3, stride=1, padding=1)
        - mu_y**2
    )

    sigma_xy = (
        F.avg_pool2d(predictions * targets, kernel_size=3, stride=1, padding=1)
        - mu_x * mu_y
    )

    ssim_numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)

    ssim_denominator = (
        (mu_x**2 + mu_y**2 + C1)
        * (sigma_x + sigma_y + C2)
    )

    ssim_score = ssim_numerator / (ssim_denominator + 1e-8)

    return 1 - ssim_score.mean()


class L1Loss(nn.Module):
    def forward(self, predictions, targets):
        return F.l1_loss(predictions, targets)


class L2Loss(nn.Module):
    def forward(self, predictions, targets):
        return F.mse_loss(predictions, targets)


class SSIMLoss(nn.Module):
    def forward(self, predictions, targets):
        return ssim_loss(predictions, targets)


class SSIML1Loss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha

    def forward(self, predictions, targets):
        loss_ssim = ssim_loss(predictions, targets)
        loss_l1 = F.l1_loss(predictions, targets)

        return self.alpha * loss_ssim + (1 - self.alpha) * loss_l1


def get_loss(loss_name: str):
    losses = {
        "l1": L1Loss,
        "l2": L2Loss,
        "ssim": SSIMLoss,
        "ssim_l1": SSIML1Loss,
    }

    if loss_name not in losses:
        raise ValueError(f"Loss no reconocida: {loss_name}")

    return losses[loss_name]()