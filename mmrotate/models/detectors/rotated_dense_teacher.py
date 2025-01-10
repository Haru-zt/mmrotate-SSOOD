from typing import Optional, Tuple

import torch
import torch.futures
import torch.nn.functional as F
from torch import Tensor

from mmengine.logging import MessageHub

from mmdet.models.utils import rename_loss_dict, reweight_loss_dict
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.losses import SmoothL1Loss

from mmrotate.models.losses import RotatedIoULoss
from mmrotate.registry import MODELS
from mmrotate.models.detectors.semi_base import RotatedSemiBaseDetector


@MODELS.register_module()
class RotatedDenseTeacher(RotatedSemiBaseDetector):
    r"""Implementation of `Dense Teacher: Dense Pseudo-Labels for
    Semi-supervised Object Detection <https://arxiv.org/abs/2207.02541v2>`_
    Args:
        detector (:obj:`ConfigDict` or dict): The detector config.
        semi_train_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised training config.
        semi_test_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised testing config.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        detector: ConfigType,
        semi_train_cfg: OptConfigType = None,
        semi_test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(
            detector=detector,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
        self.num_classes = self.student.bbox_head.cls_out_channels

        self.loss_cls_weight = self.semi_train_cfg.get("cls_weight", 1.0)

        self.bbox_loss_type = self.semi_train_cfg.get("bbox_loss_type", "l1")
        assert self.bbox_loss_type in ["RotatedIoULoss", "l1"]
        if self.bbox_loss_type == "RotatedIoULoss":
            self.loss_bbox = RotatedIoULoss(
                reduction="mean",
                loss_weight=self.semi_train_cfg.get("bbox_weight", 1.0),
            )
        elif self.bbox_loss_type == "l1":
            self.loss_bbox = SmoothL1Loss(
                reduction="none",
                loss_weight=self.semi_train_cfg.get("bbox_weight", 1.0),
            )

    @torch.no_grad()
    def get_pseudo_instances(
        self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[SampleList, Optional[dict]]:
        """Get pseudo instances from teacher model."""
        dense_predicts = self.teacher(batch_inputs)
        batch_info = {}
        batch_info["dense_predicts"] = dense_predicts
        return batch_data_samples, batch_info

    def loss_by_pseudo_instances(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList,
        batch_info: Optional[dict] = None,
    ) -> dict:
        """Calculate losses from a batch of inputs and pseudo data samples.
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`,
                which are `pseudo_instance` or `pseudo_panoptic_seg`
                or `pseudo_sem_seg` in fact.
            batch_info (dict): Batch information of teacher model
                forward propagation process. Defaults to None.
        Returns:
            dict: A dictionary of loss components
        """
        (
            teacher_cls_scores_logits,
            teacher_bbox_preds,
            teacher_angle_pred,
            teacher_centernesses,
        ) = batch_info["dense_predicts"]
        (
            student_cls_scores_logits,
            student_bbox_preds,
            student_angle_pred,
            student_centernesses,
        ) = self.student(batch_inputs)

        featmap_sizes = [featmap.size()[-2:] for featmap in teacher_cls_scores_logits]
        batch_size = teacher_cls_scores_logits[0].size(0)

        (
            flatten_teacher_cls_scores_logits,
            flatten_teacher_bbox_preds,
            flatten_teacher_centernesses,
        ) = self.flat_output(
            teacher_cls_scores_logits,
            teacher_bbox_preds,
            teacher_angle_pred,
            teacher_centernesses,
        )

        (
            flatten_student_cls_scores_logits,
            flatten_student_bbox_preds,
            flatten_student_centernesses,
        ) = self.flat_output(
            student_cls_scores_logits,
            student_bbox_preds,
            student_angle_pred,
            student_centernesses,
        )

        with torch.no_grad():
            # Region Selection according to the FSR
            ratio = self.semi_train_cfg.get("k_ratio", 0.01)
            count_num = int(teacher_cls_scores_logits.size(0) * ratio)
            teacher_probs = teacher_cls_scores_logits.sigmoid()
            max_vals = torch.max(teacher_probs, 1)[0]
            sorted_vals, sorted_inds = torch.topk(
                max_vals, teacher_cls_scores_logits.size(0)
            )
            mask = torch.zeros_like(max_vals)
            mask[sorted_inds[:count_num]] = 1.0
            fg_num = sorted_vals[:count_num].sum()

            message_hub = MessageHub.get_current_instance()
            message_hub.update_scalar("train/fg_num", fg_num)

            b_mask = mask > 0.0

        loss_cls = (
            self.loss_cls_weight
            * self.QFLv2(
                flatten_student_cls_scores_logits.sigmoid(),
                flatten_teacher_cls_scores_logits.sigmoid(),
                weight=mask,
                reduction="sum",
            )
            / fg_num
        )

        if self.bbox_loss_type == "RotatedIoULoss":
            all_level_points = self.student.bbox_head.prior_generator.grid_priors(
                featmap_sizes,
                dtype=student_bbox_preds[0].dtype,
                device=student_bbox_preds[0].device,
            )
            flatten_points = torch.cat(
                [points.repeat(batch_size, 1) for points in all_level_points]
            )

            # note that we ignore the angle_coder as PseudoAngleCoder is used
            student_bbox_preds_pos = self.student.bbox_head.bbox_coder.decode(
                flatten_points[b_mask], flatten_student_bbox_preds[b_mask]
            )
            teacher_bbox_preds_pos = self.teacher.bbox_head.bbox_coder.decode(
                flatten_points[b_mask], flatten_teacher_bbox_preds[b_mask]
            )

            # centerness weighted iou loss
            centerness_denorm = max(
                flatten_teacher_centernesses[b_mask].sigmoid().sum().detach(),
                1e-6,
            )

            loss_bbox = self.loss_bbox(
                student_bbox_preds_pos,
                teacher_bbox_preds_pos,
                weight=flatten_teacher_centernesses[b_mask].sigmoid().squeeze(-1),
                avg_factor=centerness_denorm,
            )
        elif self.bbox_loss_type == "l1":
            loss_bbox = (
                self.loss_bbox(
                    flatten_student_bbox_preds[b_mask],
                    flatten_teacher_bbox_preds[b_mask],
                )
                * flatten_teacher_centernesses[b_mask].sigmoid()
            ).mean()

        loss_centerness = F.binary_cross_entropy(
            student_centernesses[b_mask].sigmoid(),
            teacher_centernesses[b_mask].sigmoid(),
            reduction="mean",
        )

        losses = {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_centerness": loss_centerness,
        }

        unsup_weight = self.semi_train_cfg.get("unsup_weight", 1.0)
        burn_in_steps = self.semi_train_cfg.get("burn_in_steps", 10000)

        # apply burnin strategy to reweight the unsupervised weights
        target = burn_in_steps * 2
        if self.iter_count <= target:
            unsup_weight *= (self.iter_count - burn_in_steps) / burn_in_steps

        return rename_loss_dict("unsup_", reweight_loss_dict(losses, unsup_weight))

    def flat_output(self, _cls_scores_logits, _bbox_preds, _angle_pred, _centernesses):
        batch_size = _cls_scores_logits[0].size(0)
        flatten_cls_scores_logits = torch.cat(
            [
                x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
                for x in _cls_scores_logits
            ],
            dim=1,
        ).view(-1, self.num_classes)
        flatten_bbox_preds = torch.cat(
            [
                torch.cat([x, y], dim=1).permute(0, 2, 3, 1).reshape(batch_size, -1, 5)
                for x, y in zip(_bbox_preds, _angle_pred)
            ],
            dim=1,
        ).view(-1, 5)
        flatten_centernesses = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(batch_size, -1, 1) for x in _centernesses],
            dim=1,
        ).view(-1, 1)

        return flatten_cls_scores_logits, flatten_bbox_preds, flatten_centernesses

    def QFLv2(
        self, pred_sigmoid, teacher_sigmoid, weight=None, beta=2.0, reduction="mean"
    ):
        # all goes to 0
        pt = pred_sigmoid
        zerolabel = pt.new_zeros(pt.shape)
        loss = F.binary_cross_entropy(
            pred_sigmoid, zerolabel, reduction="none"
        ) * pt.pow(beta)
        pos = weight > 0

        # positive goes to bbox quality
        pt = teacher_sigmoid[pos] - pred_sigmoid[pos]
        loss[pos] = F.binary_cross_entropy(
            pred_sigmoid[pos], teacher_sigmoid[pos], reduction="none"
        ) * pt.pow(beta)

        valid = weight >= 0
        if reduction == "mean":
            loss = loss[valid].mean()
        elif reduction == "sum":
            loss = loss[valid].sum()
        return loss
