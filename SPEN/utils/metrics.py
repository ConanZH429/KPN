import torch

import torch.nn.functional as F
from torch import linalg as LA

from torch import Tensor
from torchmetrics import Metric
from typing import List


class LossMetric(Metric):
    """
    Loss metric
    self.loss is the sum of the loss
    self.num_samples is the number of samples
    """
    is_differentiable = True

    def __init__(self):
        super().__init__()
        self.add_state("loss", default=torch.tensor(0.0))
        self.add_state("num_samples", default=torch.tensor(0.0))
    
    def update(self, loss: Tensor, num_samples: int):
        self.loss += loss * num_samples
        self.num_samples += num_samples
    
    def compute(self):
        return self.loss / self.num_samples


class PosErrorMetric(Metric):
    """
    Position error metric
    self.pos_error is the sum of the position error
    self.num_samples is the number of samples
    """
    is_differentiable =True

    def __init__(self):
        super().__init__()
        self.add_state("pos_error", default=torch.tensor(0.0))
        self.add_state("Et", default=torch.tensor(0.0))
        self.add_state("num_samples", default=torch.tensor(0.0))
    
    def update(self, pos_pre: Tensor, pos_label: Tensor, num_samples: int):
        Et = LA.vector_norm(pos_pre - pos_label, dim=1)
        Et_norm = Et / LA.vector_norm(pos_label, dim=1)
        self.Et += torch.sum(Et[Et_norm>=0.002173])
        self.pos_error += torch.sum( Et_norm[Et_norm >= 0.002173] )
        self.num_samples += num_samples
        if torch.isnan(self.pos_error):
            print(pos_pre, pos_label)
    
    def compute(self):
        return self.pos_error / self.num_samples, self.Et / self.num_samples


class OriErrorMetric(Metric):
    """
    Orientation error metric
    self.ori_error is the sum of the orientation error
    self.num_samples is the number of samples
    """
    is_differentiable = True

    def __init__(self):
        super().__init__()
        self.add_state("ori_error", default=torch.tensor(0.0))
        self.add_state("num_samples", default=torch.tensor(0.0))
    
    def update(self, ori_pre: Tensor, ori_label: Tensor, num_samples: int):
        ori_pre_norm = F.normalize(ori_pre, p=2, dim=1)
        ori_label_norm = F.normalize(ori_label, p=2, dim=1)
        ori_inner_dot = torch.abs(torch.sum(ori_pre_norm * ori_label_norm, dim=1))
        ori_inner_dot = torch.clamp(ori_inner_dot, max=1.0, min=-1.0)
        ori_error = 2 * torch.rad2deg(torch.arccos(ori_inner_dot))
        self.ori_error += torch.sum(ori_error[ori_error >= 0.169])
        self.num_samples += num_samples
        if torch.isnan(self.ori_error):
            print(ori_inner_dot)
    
    def compute(self):
        return self.ori_error / self.num_samples


class KeypointsErrorMetric(Metric):
    is_differentiable = True

    def __init__(self):
        super().__init__()
        self.add_state("keypoints_error", default=torch.tensor(0.0))
        self.add_state("num_samples", default=torch.tensor(0.0))
    
    def update(
            self,
            keypoints_decode_pre: Tensor,
            keypoints_decode_label: Tensor,
            points_vis: Tensor,
            num_samples: int
        ):
        """
        keypoints_decode_pre: (B, N, 2)
        keypoints_decode_label: (B, N, 2)
        """
        keypoints_error = torch.norm(keypoints_decode_pre - keypoints_decode_label, p=2, dim=-1)  # (B, N)
        keypoints_error = keypoints_error * points_vis.float()
        keypoints_error = torch.mean(keypoints_error, dim=-1)    # (B,)
        self.keypoints_error += torch.sum(keypoints_error)
        self.num_samples += num_samples
    
    def compute(self):
        if self.num_samples == 0:
            return torch.tensor(0.0)
        return self.keypoints_error / self.num_samples


class ScoreMetric(Metric):
    is_differentiable = False

    def __init__(self, ALPHA: List[float]):
        super().__init__()
        self.add_state("score", default=torch.tensor(0.0))
        self.ALPHA = ALPHA
    
    def update(self, pos_error: Tensor, ori_error: Tensor):
        self.score = self.ALPHA[0] * pos_error + self.ALPHA[1] * ori_error
    
    def compute(self):
        return self.score