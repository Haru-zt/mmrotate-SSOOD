# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import numpy as np
import torch

from mmrotate.structures.bbox.box_converters import qbox2rbox, rbox2qbox


def norm_angle(angle, angle_range):
    """Limit the range of angles.

    Args:
        angle (ndarray): shape(n, ).
        angle_range (Str): angle representations.

    Returns:
        angle (ndarray): shape(n, ).
    """
    if angle_range == 'oc':
        return angle
    elif angle_range == 'le135':
        return (angle + np.pi / 4) % np.pi - np.pi / 4
    elif angle_range == 'le90':
        return (angle + np.pi / 2) % np.pi - np.pi / 2
    elif angle_range == 'r360':
        return (angle + np.pi) % (2 * np.pi) - np.pi
    else:
        print('Not yet implemented.')


def gaussian2bbox(gmm):
    """Convert Gaussian distribution to polygons by SVD.

    Args:
        gmm (dict[str, torch.Tensor]): Dict of Gaussian distribution.

    Returns:
        torch.Tensor: Polygons.
    """
    try:
        from torch_batch_svd import svd
    except ImportError:
        svd = None
    L = 3
    var = gmm.var
    mu = gmm.mu
    assert mu.size()[1:] == (1, 2)
    assert var.size()[1:] == (1, 2, 2)
    T = mu.size()[0]
    var = var.squeeze(1)
    if svd is None:
        raise ImportError('Please install torch_batch_svd first.')
    U, s, Vt = svd(var)
    size_half = L * s.sqrt().unsqueeze(1).repeat(1, 4, 1)
    mu = mu.repeat(1, 4, 1)
    dx_dy = size_half * torch.tensor([[-1, 1], [1, 1], [1, -1], [-1, -1]],
                                     dtype=torch.float32,
                                     device=size_half.device)
    bboxes = (mu + dx_dy.matmul(Vt.transpose(1, 2))).reshape(T, 8)

    return bboxes


def gt2gaussian(target):
    """Convert polygons to Gaussian distributions.

    Args:
        target (torch.Tensor): Polygons with shape (N, 8).

    Returns:
        dict[str, torch.Tensor]: Gaussian distributions.
    """
    L = 3
    center = torch.mean(target, dim=1)
    edge_1 = target[:, 1, :] - target[:, 0, :]
    edge_2 = target[:, 2, :] - target[:, 1, :]
    w = (edge_1 * edge_1).sum(dim=-1, keepdim=True)
    w_ = w.sqrt()
    h = (edge_2 * edge_2).sum(dim=-1, keepdim=True)
    diag = torch.cat([w, h], dim=-1).diag_embed() / (4 * L * L)
    cos_sin = edge_1 / w_
    neg = torch.tensor([[1, -1]], dtype=torch.float32).to(cos_sin.device)
    R = torch.stack([cos_sin * neg, cos_sin[..., [1, 0]]], dim=-2)

    return (center, R.matmul(diag).matmul(R.transpose(-1, -2)))


def distance2obb(points: torch.Tensor,
                 distance: torch.Tensor,
                 angle_version: str = 'oc'):
    """Convert distance angle to rotated boxes.

    Args:
        points (Tensor): Shape (B, N, 2) or (N, 2).
        distance (Tensor): Distance from the given point to 4
            boundaries and angle (left, top, right, bottom, angle).
            Shape (B, N, 5) or (N, 5)
        angle_version: angle representations.
    Returns:
        dict[str, torch.Tensor]: Gaussian distributions.
    """
    distance, angle = distance.split([4, 1], dim=-1)

    cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)

    rot_matrix = torch.cat([cos_angle, -sin_angle, sin_angle, cos_angle],
                           dim=-1)
    rot_matrix = rot_matrix.reshape(*rot_matrix.shape[:-1], 2, 2)

    wh = distance[..., :2] + distance[..., 2:]
    offset_t = (distance[..., 2:] - distance[..., :2]) / 2
    offset_t = offset_t.unsqueeze(-1)
    offset = torch.matmul(rot_matrix, offset_t).squeeze(-1)
    ctr = points[..., :2] + offset

    angle_regular = norm_angle(angle, angle_version)
    return torch.cat([ctr, wh, angle_regular], dim=-1)


def bbox_project(
    bboxes: Union[torch.Tensor, np.ndarray],
    homography_matrix: Union[torch.Tensor, np.ndarray],
    img_shape: Optional[Tuple[int, int]] = None
) -> Union[torch.Tensor, np.ndarray]:
    """Geometric transformation for bbox.

    Args:
        bboxes (Union[torch.Tensor, np.ndarray]): Shape (n, 4) for bboxes.
        homography_matrix (Union[torch.Tensor, np.ndarray]):
            Shape (3, 3) for geometric transformation.
        img_shape (Tuple[int, int], optional): Image shape. Defaults to None.
    Returns:
        Union[torch.Tensor, np.ndarray]: Converted bboxes.
    """
    bboxes_type = type(bboxes)
    if bboxes_type is np.ndarray:
        bboxes = torch.from_numpy(bboxes)
    if isinstance(homography_matrix, np.ndarray):
        homography_matrix = torch.from_numpy(homography_matrix)
    """
    fix
    """
    assert bboxes.shape[1] in [4, 5]
    if bboxes.shape[1] == 5:
        corners = rbox2qbox(bboxes).reshape(-1, 2)
    elif bboxes.shape[1] == 4:
        corners = bbox2corner(bboxes)
    corners = torch.cat(
        [corners, corners.new_ones(corners.shape[0], 1)], dim=1)
    corners = torch.matmul(homography_matrix, corners.t()).t()
    # Convert to homogeneous coordinates by normalization
    corners = corners[:, :2] / corners[:, 2:3]

    """
    fix
    """

    if bboxes.shape[1] == 5:
        corners = corners.reshape(-1, 8)
        corners[:, 0::2] = corners[:, 0::2].clamp(0, img_shape[1])
        corners[:, 1::2] = corners[:, 1::2].clamp(0, img_shape[0])
        bboxes = qbox2rbox(corners)
    elif bboxes.shape[1] == 4:
        bboxes = corner2bbox(corners)
        if img_shape is not None:
            bboxes[:, 0::2] = bboxes[:, 0::2].clamp(0, img_shape[1])
            bboxes[:, 1::2] = bboxes[:, 1::2].clamp(0, img_shape[0])
    # fix problem
    # if img_shape is not None:
    #     bboxes[:, 0::2] = bboxes[:, 0::2].clamp(0, img_shape[1])
    #     bboxes[:, 1::2] = bboxes[:, 1::2].clamp(0, img_shape[0])
    if bboxes_type is np.ndarray:
        bboxes = bboxes.numpy()
    return bboxes

def bbox2corner(bboxes: torch.Tensor) -> torch.Tensor:
    """Convert bbox coordinates from (x1, y1, x2, y2) to corners ((x1, y1),
    (x2, y1), (x1, y2), (x2, y2)).

    Args:
        bboxes (Tensor): Shape (n, 4) for bboxes.
    Returns:
        Tensor: Shape (n*4, 2) for corners.
    """
    x1, y1, x2, y2 = torch.split(bboxes, 1, dim=1)
    return torch.cat([x1, y1, x2, y1, x1, y2, x2, y2], dim=1).reshape(-1, 2)

def corner2bbox(corners: torch.Tensor) -> torch.Tensor:
    """Convert bbox coordinates from corners ((x1, y1), (x2, y1), (x1, y2),
    (x2, y2)) to (x1, y1, x2, y2).

    Args:
        corners (Tensor): Shape (n*4, 2) for corners.
    Returns:
        Tensor: Shape (n, 4) for bboxes.
    """
    corners = corners.reshape(-1, 4, 2)
    min_xy = corners.min(dim=1)[0]
    max_xy = corners.max(dim=1)[0]
    return torch.cat([min_xy, max_xy], dim=1)
