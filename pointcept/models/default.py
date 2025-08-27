import torch
import torch.nn as nn
import torch_scatter
import torch_cluster

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DINOEnhancedSegmentor(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone) if backbone is not None else None
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.backbone is not None and self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.backbone is not None:
            if self.freeze_backbone:
                with torch.no_grad():
                    point = self.backbone(point)
            else:
                point = self.backbone(point)
            point_list = [point]
            while "unpooling_parent" in point_list[-1].keys():
                point_list.append(point_list[-1].pop("unpooling_parent"))
            for i in reversed(range(1, len(point_list))):
                point = point_list[i]
                parent = point_list[i - 1]
                assert "pooling_inverse" in point.keys()
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = point_list[0]
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = [point.feat]
        else:
            feat = []
        dino_coord = input_dict["dino_coord"]
        dino_feat = input_dict["dino_feat"]
        dino_offset = input_dict["dino_offset"]
        idx = torch_cluster.knn(
            x=dino_coord,
            y=point.origin_coord,
            batch_x=offset2batch(dino_offset),
            batch_y=offset2batch(point.origin_offset),
            k=1,
        )[1]

        feat.append(dino_feat[idx])
        feat = torch.concatenate(feat, dim=-1)
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)



@MODELS.register_module()
class PTV3WithDecoder(nn.Module):
    """
    PTv3 backbone + simple linear semantic head + external decoder producing per-class query masks.

    TRAIN (self.training=True): return ONLY scalars at top-level:
        {"loss": <scalar>, "seg_loss": <scalar>, "decoder_loss": <scalar>}
    EVAL (has GT, not training): {"loss": <scalar>, "seg_logits": (N,K)}
    TEST (no GT): {"seg_logits": (N,K)}
    """

    def __init__(
        self,
        backbone,
        decoder,
        num_classes,
        backbone_out_channels: int = 64,
        criteria=None,                 # CE + Lovasz (list-of-losses is fine via build_criteria)
        decoder_criteria=None,         # e.g., SemanticMaskBCELoss
        decoder_loss_weight: float = 1.0,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.decoder  = build_model(decoder)
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

        self.criteria = build_criteria(criteria) if criteria is not None else None
        self.decoder_criteria = (
            build_criteria(decoder_criteria) if decoder_criteria is not None else None
        )
        self.decoder_loss_weight = float(decoder_loss_weight)

        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @staticmethod
    def _offset_cum2sizes(offset_cum: torch.Tensor):
        sizes, prev = [], 0
        for v in offset_cum.detach().cpu().tolist():
            sizes.append(int(v - prev)); prev = int(v)
        return sizes

    def _stitch_feats(self, point: Point):
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent  = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            return point.feat, point
        else:
            return point, None

    def _make_feats_list(self, feat: torch.Tensor, input_dict, point_obj: Point):
        offset_cum = point_obj.offset if (point_obj is not None and hasattr(point_obj, "offset")) else input_dict["offset"]
        sizes = self._offset_cum2sizes(offset_cum)
        return list(torch.split(feat, sizes, dim=0)), offset_cum

    def forward(self, input_dict):
        # Backbone
        point = Point(input_dict)
        if self.freeze_backbone:
            with torch.no_grad():
                point = self.backbone(point)
        else:
            point = self.backbone(point)

        # Original-res features + logits
        feat, point_obj = self._stitch_feats(point)   # (N, C)
        seg_logits = self.seg_head(feat)              # (N, K)

        # Decoder (operates per-sample list)
        feats_list, offset_cum = self._make_feats_list(feat, input_dict, point_obj)
        dec_out = self.decoder(feats_list)            # dict with "masks": List[(K, ni)]
        query_masks = dec_out.get("masks", None)

        has_gt = ("segment" in input_dict)

        # ---------------- TRAIN or EVAL (with GT) ----------------
        if self.training or has_gt:
            total = torch.zeros((), device=seg_logits.device)
            out = {}

            # semantic head loss
            if self.criteria is not None:
                seg_loss = self.criteria(seg_logits, input_dict["segment"])
                out["seg_loss"] = seg_loss
                total = total + seg_loss

            # decoder loss (list-of-masks vs list-of-targets)
            if (self.decoder_criteria is not None) and (query_masks is not None):
                tgt_flat = input_dict["segment"].long()      # (N,)
                sizes = self._offset_cum2sizes(input_dict["offset"])
                tgt_list = list(torch.split(tgt_flat, sizes, dim=0))  # List[(ni,)]
                dec_loss = self.decoder_criteria(query_masks, tgt_list)
                dec_loss = self.decoder_loss_weight * dec_loss
                out["decoder_loss"] = dec_loss
                total = total + dec_loss

            out["loss"] = total

            if self.training:
                # IMPORTANT: top-level MUST be scalars only
                return out
            else:
                # eval with GT: expose logits for evaluator
                out["seg_logits"] = seg_logits
                return out

        # ---------------- TEST (no GT) ----------------
        return {"seg_logits": seg_logits}
