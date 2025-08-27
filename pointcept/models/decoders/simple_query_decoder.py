import torch
import torch.nn as nn
from typing import List, Dict, Union

from pointcept.models.builder import MODELS


@MODELS.register_module()
class SimpleQueryDecoder(nn.Module):
    """
    Minimal training-friendly query decoder for Pointcept.

    Inputs (choose one API):
      1) x_list: List[Tensor], each of shape (Ni, in_channels)
      2) x + offset: x is (N, in_channels) stacked over batch; offset is cumulative CSR (B,)

    Returns (logits, no activation):
      {
        "masks":  List[Tensor], each (K, Ni)  -- per-class mask logits for each sample
        # present only if use_score_head=True:
        "scores": List[Tensor], each (K, 1)   -- per-class query/objectness logits
        # optional:
        "point_logits": List[Tensor], each (Ni, K) -- if return_point_logits=True
      }

    DDP-safety:
      - By default, the score head is DISABLED (use_score_head=False), so no unused
        learnable parameters exist that are not part of your loss.
      - If you enable use_score_head=True but do NOT include it in any loss, set
        train_score_head=False (default), which freezes those params to avoid
        DDP "unused parameter" reduction errors.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        num_classes: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        return_point_logits: bool = False,
        # new flags:
        use_score_head: bool = False,     # create score head?
        train_score_head: bool = False,   # if created, should it require grad?
    ):
        super().__init__()
        self.num_classes = num_classes
        self.return_point_logits = return_point_logits

        # encoder over point features
        self.in_proj = nn.Linear(in_channels, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # queries (one per semantic class)
        self.queries = nn.Embedding(num_classes, d_model)
        self.q_norm = nn.LayerNorm(d_model)

        # mask projection (used for per-class point masks)
        self.mask_proj = nn.Linear(d_model, d_model)

        # optional score head
        self.use_score_head = bool(use_score_head)
        self.train_score_head = bool(train_score_head)
        if self.use_score_head:
            self.score = nn.Linear(d_model, 1)
            if not self.train_score_head:
                for p in self.score.parameters():
                    p.requires_grad = False
        else:
            self.score = None  # not registered -> no params -> no DDP issues

    @staticmethod
    def _split_by_offset(feat: torch.Tensor, offset: torch.Tensor) -> List[torch.Tensor]:
        """Split a (N, C) stacked tensor into a list using cumulative CSR offsets."""
        sizes, prev = [], 0
        for v in offset.tolist():
            sizes.append(v - prev)
            prev = v
        return list(torch.split(feat, sizes, dim=0))

    def _forward_single(self, x_i: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Single-sample forward.
        x_i: (Ni, in_channels)
        Returns dict with:
          masks: (K, Ni) logits
          scores: (K, 1) logits (only if score head is enabled)
          point_logits: (Ni, K) logits (optional)
        """
        # per-point features
        h = self.in_proj(x_i)            # (Ni, d)
        h = self.encoder(h.unsqueeze(0)) # (1, Ni, d)
        h = h.squeeze(0)                 # (Ni, d)

        # queries
        Q = self.queries.weight.to(h.dtype).to(h.device)  # (K, d)
        Qn = self.q_norm(Q)                               # (K, d)

        # per-class mask logits over points: (K, Ni) = (K, d) @ (d, Ni)
        q_proj = self.mask_proj(Qn)                       # (K, d)
        masks = torch.matmul(q_proj, h.transpose(0, 1))   # (K, Ni)

        out = {"masks": masks}

        if self.use_score_head and (self.score is not None):
            scores = self.score(Qn)                       # (K, 1)
            out["scores"] = scores

        if self.return_point_logits:
            point_logits = masks.transpose(0, 1).contiguous()  # (Ni, K)
            out["point_logits"] = point_logits

        return out

    def forward(
        self,
        x: Union[List[torch.Tensor], torch.Tensor],
        offset: torch.Tensor = None,
        queries: List[torch.Tensor] = None,  # reserved for future per-sample dynamic queries
    ) -> Dict[str, List[torch.Tensor]]:
        # normalize to list API
        if isinstance(x, list):
            x_list = x
        else:
            assert offset is not None, "Provide 'offset' when passing a single stacked (N, C) tensor."
            x_list = self._split_by_offset(x, offset)

        out_lists: Dict[str, List[torch.Tensor]] = {"masks": []}
        if self.use_score_head and (self.score is not None):
            out_lists["scores"] = []
        if self.return_point_logits:
            out_lists["point_logits"] = []

        for x_i in x_list:
            single = self._forward_single(x_i)
            out_lists["masks"].append(single["masks"])
            if "scores" in single:
                out_lists["scores"].append(single["scores"])
            if "point_logits" in single:
                out_lists["point_logits"].append(single["point_logits"])

        return out_lists
