from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from mmengine.logging import MessageHub

from mmdet.models.utils import rename_loss_dict, reweight_loss_dict
from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.losses import CrossEntropyLoss, SmoothL1Loss

from mmrotate.registry import MODELS
from mmrotate.models.detectors.semi_base import RotatedSemiBaseDetector


@MODELS.register_module()
class RotatedDenserTeacher(RotatedSemiBaseDetector):
    """Implementation of `Denser Teacher: Rethinking Dense Pseudo-Label for Semi-Supervised Oriented Object Detection`
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
        assert self.bbox_loss_type in ["l1"]
        if self.bbox_loss_type == "l1":
            self.loss_bbox = SmoothL1Loss(
                reduction="none",
                loss_weight=self.semi_train_cfg.get("bbox_weight", 1.0),
            )

        self.centerness_loss_type = self.semi_train_cfg.get(
            "centerness_loss_type", "BCE"
        )
        assert self.centerness_loss_type in ["BCE", "SmoothL1"]

        if self.centerness_loss_type == "BCE":
            # use_sigmoid=True for BCEWithLogits
            # not use in fact
            self.loss_centerness = CrossEntropyLoss(
                use_sigmoid=True,
                reduction="none",
                loss_weight=self.semi_train_cfg.get("centerness_weight", 1.0),
            )
        elif self.centerness_loss_type == "SmoothL1":
            self.loss_centerness = SmoothL1Loss(
                reduction="mean",
                loss_weight=self.semi_train_cfg.get("centerness_weight", 1.0),
            )

        self.loss_dense_msl_weight = self.semi_train_cfg.get("dense_msl_weight", 1.0)

        self.prior_generator = MlvlPointGenerator(
            strides=[8, 16, 32, 64, 128], offset=0.5
        )

        self.dense_msl = self.semi_train_cfg.get("dense_msl", False)

        if self.dense_msl:
            self.loss_bbox_resize = SmoothL1Loss(
                reduction="none",
                loss_weight=self.semi_train_cfg.get("bbox_weight", 1.0),
            )
            self.loss_centerness_resize = SmoothL1Loss(
                reduction="mean",
                loss_weight=self.semi_train_cfg.get("centerness_weight", 1.0),
            )

    @torch.no_grad()
    def get_pseudo_instances(
        self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[SampleList, Optional[dict]]:
        """Get pseudo instances from teacher model."""
        dense_predicts = self.teacher(batch_inputs)
        batch_info = {}
        batch_info["dense_predicts"] = dense_predicts
        if self.dense_msl:
            batch_inputs_resize = F.interpolate(
                batch_inputs, scale_factor=0.5, mode="nearest"
            )
            dense_predicts_resize = self.teacher(batch_inputs_resize)
            batch_info["dense_predicts_resize"] = dense_predicts_resize
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
            teacher_probs = flatten_teacher_cls_scores_logits.sigmoid()
            max_vals = torch.max(teacher_probs, 1)[0]
            avg_fsr = max_vals.mean()
            avg_fsr = self.semi_train_cfg.get("fsr_weight", 1.0) * avg_fsr

            if self.dense_msl:
                (
                    teacher_cls_scores_logits_resize,
                    teacher_bbox_preds_resize,
                    teacher_angle_pred_resize,
                    teacher_centernesses_resize,
                ) = batch_info["dense_predicts_resize"]
                (
                    flatten_teacher_cls_scores_logits_resize,
                    flatten_teacher_bbox_preds_resize,
                    flatten_teacher_centernesses_resize,
                ) = self.flat_output(
                    teacher_cls_scores_logits_resize,
                    teacher_bbox_preds_resize,
                    teacher_angle_pred_resize,
                    teacher_centernesses_resize,
                )
                teacher_probs_resize = (
                    flatten_teacher_cls_scores_logits_resize.sigmoid()
                )
                max_vals_resize = torch.max(teacher_probs_resize, 1)[0]
                avg_fsr_resize = max_vals_resize.mean()
                avg_fsr_resize = (
                    self.semi_train_cfg.get("fsr_weight", 1.0) * avg_fsr_resize
                )

            count_num = int(flatten_teacher_cls_scores_logits.size(0) * avg_fsr)

            sorted_vals, sorted_inds = torch.topk(
                max_vals, flatten_teacher_cls_scores_logits.size(0)
            )
            mask = torch.zeros_like(max_vals)
            mask[sorted_inds[:count_num]] = 1.0
            fg_num = sorted_vals[:count_num].sum()

            if self.dense_msl:
                count_num_resize = int(
                    flatten_teacher_cls_scores_logits_resize.size(0) * avg_fsr_resize
                )
                sorted_vals_resize, sorted_inds_resize = torch.topk(
                    max_vals_resize,
                    flatten_teacher_cls_scores_logits_resize.size(0),
                )
                mask_resize = torch.zeros_like(max_vals_resize)
                mask_resize[sorted_inds_resize[:count_num_resize]] = 1.0
                fg_num_resize = sorted_vals_resize[:count_num_resize].sum()

        message_hub = MessageHub.get_current_instance()
        message_hub.update_scalar("train/K", avg_fsr)
        message_hub.update_scalar("train/count_num", count_num)
        message_hub.update_scalar("train/fg_num", fg_num)
        b_mask = mask > 0.0

        if self.dense_msl:
            b_mask_resize = mask_resize > 0.0
            message_hub.update_scalar("train/K_resize", avg_fsr_resize)
            message_hub.update_scalar("train/count_num_resize", count_num_resize)
            message_hub.update_scalar("train/fg_num_resize", fg_num_resize)

        # if avg_fsr == 0.0:
        if avg_fsr == 0.0 or fg_num == 0.0:
            loss_cls = torch.tensor(
                0.0, device=flatten_student_cls_scores_logits.device
            )
            loss_bbox = torch.tensor(
                0.0, device=flatten_student_cls_scores_logits.device
            )
            loss_centerness = torch.tensor(
                0.0, device=flatten_student_cls_scores_logits.device
            )

        else:
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

            loss_bbox = (
                self.loss_bbox(
                    flatten_student_bbox_preds[b_mask],
                    flatten_teacher_bbox_preds[b_mask],
                )
                * flatten_teacher_centernesses.sigmoid()[b_mask]
            ).mean()

            if self.centerness_loss_type == "BCE":
                loss_centerness = F.binary_cross_entropy(
                    flatten_student_centernesses[b_mask].sigmoid(),
                    flatten_teacher_centernesses[b_mask].sigmoid(),
                    reduction="mean",
                )
            elif self.centerness_loss_type == "SmoothL1":
                loss_centerness = self.loss_centerness(
                    flatten_student_centernesses[b_mask].sigmoid(),
                    flatten_teacher_centernesses[b_mask].sigmoid(),
                )

        if self.dense_msl:
            if avg_fsr_resize == 0.0 or fg_num_resize == 0.0:
                loss_dense_msl_cls = torch.tensor(
                    0.0, device=flatten_student_cls_scores_logits.device
                )
                loss_dense_msl_bbox = torch.tensor(
                    0.0, device=flatten_student_cls_scores_logits.device
                )
                loss_dense_msl_centerness = torch.tensor(
                    0.0, device=flatten_student_cls_scores_logits.device
                )
            else:
                batch_inputs_resize = F.interpolate(
                    batch_inputs, scale_factor=0.5, mode="nearest"
                )
                (
                    student_cls_scores_logits_resize,
                    student_bbox_preds_resize,
                    student_angle_pred_resize,
                    student_centernesses_resize,
                ) = self.student(batch_inputs_resize)
                (
                    flatten_student_cls_scores_logits_resize,
                    flatten_student_bbox_preds_resize,
                    flatten_student_centernesses_resize,
                ) = self.flat_output(
                    student_cls_scores_logits_resize,
                    student_bbox_preds_resize,
                    student_angle_pred_resize,
                    student_centernesses_resize,
                )

                loss_dense_msl_cls = (
                    self.loss_dense_msl_weight
                    * self.QFLv2(
                        flatten_student_cls_scores_logits_resize.sigmoid(),
                        flatten_teacher_cls_scores_logits_resize.sigmoid(),
                        weight=mask_resize,
                        reduction="sum",
                    )
                    / fg_num_resize
                )

                loss_dense_msl_bbox = (
                    self.loss_dense_msl_weight
                    * (
                        self.loss_bbox_resize(
                            flatten_student_bbox_preds_resize[b_mask_resize],
                            flatten_teacher_bbox_preds_resize[b_mask_resize],
                        )
                        * flatten_teacher_centernesses_resize.sigmoid()[b_mask_resize]
                    ).mean()
                )

                if self.centerness_loss_type == "BCE":
                    loss_dense_msl_centerness = (
                        self.loss_dense_msl_weight
                        * F.binary_cross_entropy(
                            flatten_student_centernesses_resize[
                                b_mask_resize
                            ].sigmoid(),
                            flatten_teacher_centernesses_resize[
                                b_mask_resize
                            ].sigmoid(),
                            reduction="mean",
                        )
                    )
                elif self.centerness_loss_type == "SmoothL1":
                    loss_dense_msl_centerness = (
                        self.loss_dense_msl_weight
                        * self.loss_centerness_resize(
                            flatten_student_centernesses_resize[
                                b_mask_resize
                            ].sigmoid(),
                            flatten_teacher_centernesses_resize[
                                b_mask_resize
                            ].sigmoid(),
                        )
                    )

        if self.dense_msl:
            losses = {
                "loss_cls": loss_cls,
                "loss_bbox": loss_bbox,
                "loss_centerness": loss_centerness,
                "loss_dense_msl_cls": loss_dense_msl_cls,
                "loss_dense_msl_bbox": loss_dense_msl_bbox,
                "loss_dense_msl_centerness": loss_dense_msl_centerness,
            }
        else:
            losses = {
                "loss_cls": loss_cls,
                "loss_bbox": loss_bbox,
                "loss_centerness": loss_centerness,
            }

        unsup_weight = self.semi_train_cfg.get("unsup_weight", 1.0)
        burn_in_steps = self.semi_train_cfg.get("burn_in_steps", 6400)

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
