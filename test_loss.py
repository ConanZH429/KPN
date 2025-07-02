import torch

from SPEN.utils import get_uncertainty_loss

uncertainty_loss_func = get_uncertainty_loss(
    uncertainty_type="Rank",
    loss_type="Sub",
    beta=1.0,
    weight_strategy=None,
)

uncertainty_label = torch.tensor([
    [0.1, 0.2, 0.3],
    [0.1, 0.2, 0.3]
])

uncertainty_pre = torch.tensor([
    [0.1, 0.2, 0.3],
    [0.1, 0.2, 0.5]
])

uncertainty_loss = uncertainty_loss_func(
    uncertainty_pre=uncertainty_pre,
    uncertainty_label=uncertainty_label,
    now_epoch=0
)

print("Uncertainty Loss:", uncertainty_loss.item())