import torch

from src.losses import get_loss


predictions = torch.rand(8, 3, 128, 128)
targets = torch.rand(8, 3, 128, 128)

for loss_name in ["l1", "l2", "ssim", "ssim_l1"]:
    loss_fn = get_loss(loss_name)
    loss = loss_fn(predictions, targets)

    print(f"{loss_name}: {loss.item():.6f}")