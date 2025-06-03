import torch
from .config import Config

def get_optimizer(model: torch.nn.Module, optimizer_type: str, config: Config) -> torch.optim.Optimizer:
    """
    Get the optimizer from the given optimizer type

    Args:
        optimizer_type (str, optional): The optimizer type to use.
    
    Returns:
        torch.optim.Optimizer: The optimizer
    """
    lr = config.lr0 if hasattr(config, "lr0") else 0.001
    weight_decay = config.weight_decay if hasattr(config, "weight_decay") else 0.0
    momentum = config.momentum if hasattr(config, "momentum") else 0.0
    if optimizer_type == "SGD":
        lr = config.lr0 if hasattr(config, "lr0") else 0.01
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_type == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "NAdam":
        return torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")