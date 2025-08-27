# pointcept/models/losses/query_mask_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import LOSSES


@LOSSES.register_module()
class SemanticMaskBCELoss(nn.Module):
    def __init__(self, ignore_index: int = -1, loss_weight: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, **kwargs):
        # pred: List[Tensor] each (K, Ni) OR single Tensor (K, N)
        # target: List[Tensor] each (Ni,) OR single Tensor (N,)
        # (We assume model already split to a list; if it's a flat tensor,
        # you could still support it here by reading kwargs.get('offset'))
        if isinstance(pred, torch.Tensor):
            pred_list = [pred]
        else:
            pred_list = list(pred)

        if isinstance(target, (list, tuple)):
            tgt_list = list(target)
        else:
            # Fallback single-sample path
            tgt_list = [target]

        assert len(pred_list) == len(tgt_list), "pred and target must align in length"

        total = pred_list[0].new_tensor(0.0)
        total_count = 0

        for masks, tgt in zip(pred_list, tgt_list):
            # masks: (K, Ni), tgt: (Ni,)
            K, Ni = masks.shape
            if tgt.numel() != Ni:
                raise ValueError(f"Target length {tgt.numel()} != number of points {Ni}")
            valid = (tgt != self.ignore_index)
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue

            tgt_valid = tgt[valid].long()
            gt = torch.zeros((K, n_valid), dtype=masks.dtype, device=masks.device)
            gt.scatter_(0, tgt_valid.unsqueeze(0), 1.0)  # one-hot (K, n_valid)

            loss_i = F.binary_cross_entropy_with_logits(masks[:, valid], gt, reduction="mean")
            total = total + loss_i * (n_valid if self.reduction == "mean" else 1.0)
            total_count += (n_valid if self.reduction == "mean" else 1)

        if total_count == 0:
            return total  # 0.0

        if self.reduction == "mean":
            total = total / total_count

        return self.loss_weight * total
