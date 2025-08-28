import torch
import torch.nn as nn
from typing import List, Dict, Union

from pointcept.models.builder import MODELS


class _CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, ffn_mult: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        """
        q:   (B, K, d)
        mem: (B, N, d)
        """
        q2, _ = self.self_attn(q, q, q)          # self-attn over queries
        q = self.norm1(q + self.drop(q2))
        q2, _ = self.cross_attn(q, mem, mem)     # cross-attn: queries attend to points
        q = self.norm2(q + self.drop(q2))
        q2 = self.ffn(q)
        q = self.norm3(q + self.drop(q2))
        return q


@MODELS.register_module()
class SimpleQueryDecoderWithCrossAttention(nn.Module):
    """
    Cross-attention query decoder for Pointcept.

    Inputs (choose one API):
      1) x_list: List[Tensor], each (Ni, in_channels)
      2) x + offset: x is (N, in_channels) stacked over batch; offset is cumulative CSR (B,)

    Returns (logits, no activation):
      {
        "masks":        List[Tensor], each (K, Ni)   -- per-class mask logits
        "scores":       List[Tensor], each (K, 1)    -- OPTIONAL, if use_score_head=True
        "point_logits": List[Tensor], each (Ni, K)   -- OPTIONAL, if return_point_logits=True
      }

    Notes:
      - DDP-safe score head: if enabled but unused in loss, set train_score_head=False.
      - Memory encoder is a light TransformerEncoder over point features (can be disabled by num_mem_layers=0).
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        num_classes: int,
        num_layers: int = 2,            # number of cross-attn blocks
        num_heads: int = 4,
        dropout: float = 0.0,
        return_point_logits: bool = False,
        use_score_head: bool = False,
        train_score_head: bool = False,
        # optional memory (point feature) encoder:
        num_mem_layers: int = 0,
        mem_ffn_mult: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.return_point_logits = bool(return_point_logits)

        # project per-point features to model dim
        self.in_proj = nn.Linear(in_channels, d_model)

        # lightweight memory encoder over points (can be disabled via num_mem_layers=0)
        if num_mem_layers > 0:
            mem_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=mem_ffn_mult * d_model,
                dropout=dropout,
                batch_first=True,
            )
            self.mem_encoder = nn.TransformerEncoder(mem_layer, num_layers=num_mem_layers)
        else:
            self.mem_encoder = None

        # learned class queries
        self.queries = nn.Embedding(num_classes, d_model)

        # cross-attention stack
        self.blocks = nn.ModuleList([
            _CrossAttnBlock(d_model, num_heads, dropout=dropout) for _ in range(num_layers)
        ])

        # heads
        self.q_norm = nn.LayerNorm(d_model)
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
            self.score = None  # no params registered -> no DDP issues

    @staticmethod
    def _split_by_offset(feat: torch.Tensor, offset: torch.Tensor) -> List[torch.Tensor]:
        """Split a stacked (N, C) tensor into a list using cumulative CSR offsets (B,)."""
        sizes, prev = [], 0
        for v in offset.tolist():
            sizes.append(int(v - prev))
            prev = int(v)
        return list(torch.split(feat, sizes, dim=0))

    def _forward_single(self, x_i: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Single-sample forward.
        x_i: (Ni, in_channels)
        Returns:
          masks: (K, Ni)
          scores: (K, 1) if enabled
          point_logits: (Ni, K) if enabled
        """
        # point memory
        mem = self.in_proj(x_i).unsqueeze(0)     # (1, N, d)
        if self.mem_encoder is not None:
            mem = self.mem_encoder(mem)          # (1, N, d)

        # queries
        q = self.queries.weight.unsqueeze(0)     # (1, K, d)

        # cross-attention stack
        for blk in self.blocks:
            q = blk(q, mem)                      # (1, K, d)

        # heads
        q = self.q_norm(q.squeeze(0))            # (K, d)
        mem = mem.squeeze(0)                     # (N, d)
        q_mask = self.mask_proj(q)               # (K, d)

        # per-class mask logits over points: (K, Ni) = (K, d) @ (d, Ni)
        masks = torch.matmul(q_mask, mem.transpose(0, 1))  # (K, N)

        out: Dict[str, torch.Tensor] = {"masks": masks}

        if self.use_score_head and (self.score is not None):
            out["scores"] = self.score(q)        # (K, 1)

        if self.return_point_logits:
            out["point_logits"] = masks.transpose(0, 1).contiguous()  # (N, K)

        return out

    def forward(
        self,
        x: Union[List[torch.Tensor], torch.Tensor],
        offset: torch.Tensor = None,
        queries: List[torch.Tensor] = None,  # reserved for future dynamic queries
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
