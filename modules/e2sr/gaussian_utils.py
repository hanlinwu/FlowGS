"""
高斯散射工具函数模块
用于支持连续尺度超分辨率的坐标生成和特征查询

基于 GaussianSR 的实现，适配 LDCSR 第一阶段
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_coord(shape, ranges=None, flatten=True):
    """
    生成归一化坐标网格

    Args:
        shape: (H, W) - 网格尺寸
        ranges: 坐标范围列表 [(v0_h, v1_h), (v0_w, v1_w)]，默认 [(-1, 1), (-1, 1)]
        flatten: 是否展平为 [H*W, 2]

    Returns:
        coord: 坐标张量
            - flatten=True: [H*W, 2]
            - flatten=False: [H, W, 2]

    Example:
        >>> coord = make_coord((64, 64))
        >>> coord.shape
        torch.Size([4096, 2])
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)

    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def make_cell(coord, shape):
    """
    计算每个坐标点对应的像素单元大小

    Args:
        coord: [N, 2] 或 [B, N, 2] - 归一化坐标
        shape: (H, W) - 目标图像尺寸

    Returns:
        cell: 与 coord 相同形状的像素单元大小张量

    Example:
        >>> coord = make_coord((128, 128))  # [16384, 2]
        >>> cell = make_cell(coord, (128, 128))
        >>> cell[0]  # 每个像素单元在归一化空间的大小
        tensor([0.0156, 0.0156])  # 2/128 ≈ 0.0156
    """
    cell = torch.ones_like(coord)
    cell[..., 0] *= 2 / shape[0]
    cell[..., 1] *= 2 / shape[1]
    return cell


def to_pixel_samples(img):
    """
    将图像转换为像素采样格式 (坐标-颜色对)

    Args:
        img: [B, C, H, W] - 输入图像

    Returns:
        coord: [H*W, 2] - 归一化坐标 (所有批次共享)
        rgb: [B, H*W, C] - 每个坐标的颜色值

    Example:
        >>> img = torch.rand(4, 3, 64, 64)
        >>> coord, rgb = to_pixel_samples(img)
        >>> coord.shape, rgb.shape
        (torch.Size([4096, 2]), torch.Size([4, 4096, 3]))
    """
    B, C, H, W = img.shape
    coord = make_coord((H, W))  # [H*W, 2]
    rgb = img.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
    return coord, rgb


def generate_meshgrid(height, width, device='cpu'):
    """
    生成像素坐标网格 (整数坐标)

    Args:
        height: 图像高度
        width: 图像宽度
        device: 设备

    Returns:
        coords: [H*W, 2] - (x, y) 整数坐标对

    Example:
        >>> coords = generate_meshgrid(3, 3)
        >>> coords
        tensor([[0, 0], [1, 0], [2, 0],
                [0, 1], [1, 1], [2, 1],
                [0, 2], [1, 2], [2, 2]])
    """
    y_coords = torch.arange(0, height, device=device)
    x_coords = torch.arange(0, width, device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return coords


def fetching_features_from_tensor(image_tensor, input_coords):
    """
    从特征张量中提取指定坐标的特征值

    Args:
        image_tensor: [B, C, H, W] - 特征图
        input_coords: [N, 2] - 整数坐标 (x, y)

    Returns:
        color_values: [B, N, C] - 提取的特征值
        coords_norm: [N, 2] - 归一化坐标 [-1, 1]

    Example:
        >>> feat = torch.rand(2, 64, 32, 32)
        >>> coords = torch.tensor([[0, 0], [15, 15], [31, 31]])
        >>> values, coords_norm = fetching_features_from_tensor(feat, coords)
        >>> values.shape
        torch.Size([2, 3, 64])
    """
    # 归一化坐标到 [-1, 1]
    input_coords = input_coords.to(image_tensor.device)
    H, W = image_tensor.shape[-2:]

    coords = input_coords.float() / torch.tensor([H, W], device=image_tensor.device).float()
    center_coords_normalized = torch.tensor([0.5, 0.5], device=image_tensor.device).float()
    coords_norm = (center_coords_normalized - coords) * 2.0

    # 提取特征值
    batch_size = image_tensor.shape[0]
    input_coords_expanded = input_coords.unsqueeze(0).expand(batch_size, -1, -1)

    y_coords = input_coords_expanded[..., 0].long()
    x_coords = input_coords_expanded[..., 1].long()
    batch_indices = torch.arange(batch_size).view(-1, 1).to(input_coords.device)

    color_values = image_tensor[batch_indices, :, x_coords, y_coords]

    return color_values, coords_norm


def extract_patch(image, center, radius, padding_mode='constant'):
    """
    从图像中提取以 center 为中心、半径为 radius 的方形 patch

    Args:
        image: [B, C, H, W] - 输入图像
        center: (y, x) - patch 中心坐标
        radius: patch 半径 (最终大小为 2*radius × 2*radius)
        padding_mode: 填充模式 ('constant', 'reflect', 'replicate', 'circular')

    Returns:
        patch: [B, C, 2*radius, 2*radius] - 提取的 patch

    Example:
        >>> img = torch.rand(1, 3, 64, 64)
        >>> patch = extract_patch(img, center=(32, 32), radius=8)
        >>> patch.shape
        torch.Size([1, 3, 16, 16])
    """
    height, width = image.shape[-2:]

    # 转换为整数坐标
    center_y, center_x = int(round(center[0])), int(round(center[1]))

    # 计算边界
    top = center_y - radius
    bottom = center_y + radius
    left = center_x - radius
    right = center_x + radius

    # 计算需要的填充
    top_padding = max(0, -top)
    bottom_padding = max(0, bottom - height)
    left_padding = max(0, -left)
    right_padding = max(0, right - width)

    # 填充图像
    padded_image = F.pad(
        image,
        (left_padding, right_padding, top_padding, bottom_padding),
        mode=padding_mode
    )

    # 提取 patch
    patch = padded_image[
        ...,
        top_padding:top_padding + 2 * radius,
        left_padding:left_padding + 2 * radius
    ]

    return patch


def sample_random_coords(coord, num_samples):
    """
    从坐标集合中随机采样

    Args:
        coord: [N, 2] - 坐标张量
        num_samples: 采样数量

    Returns:
        sampled_coord: [num_samples, 2] - 采样的坐标
        sampled_indices: [num_samples] - 采样的索引

    Example:
        >>> coord = make_coord((64, 64))  # [4096, 2]
        >>> sampled, indices = sample_random_coords(coord, 2304)
        >>> sampled.shape
        torch.Size([2304, 2])
    """
    num_total = coord.shape[0]
    if num_samples >= num_total:
        return coord, torch.arange(num_total)

    indices = torch.randperm(num_total)[:num_samples]
    sampled_coord = coord[indices]
    return sampled_coord, indices


def coords_to_pixel_indices(coords, shape):
    """
    将归一化坐标转换为像素索引

    Args:
        coords: [N, 2] - 归一化坐标 [-1, 1]
        shape: (H, W) - 图像尺寸

    Returns:
        indices: [N, 2] - 像素索引 (y, x)

    Example:
        >>> coords = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        >>> indices = coords_to_pixel_indices(coords, (64, 64))
        >>> indices
        tensor([[ 0,  0],
                [32, 32],
                [63, 63]])
    """
    H, W = shape
    # 从 [-1, 1] 转换到 [0, H-1] 和 [0, W-1]
    y = ((coords[:, 0] + 1) / 2 * (H - 1)).round().long()
    x = ((coords[:, 1] + 1) / 2 * (W - 1)).round().long()
    indices = torch.stack([y, x], dim=1)
    return indices


def create_coord_feat(feat_shape, device='cpu'):
    """
    创建与特征图对应的坐标特征图

    Args:
        feat_shape: (B, C, H, W) - 特征图尺寸
        device: 设备

    Returns:
        coord_feat: [B, 2, H, W] - 坐标特征图

    Example:
        >>> coord_feat = create_coord_feat((4, 64, 32, 32), 'cuda')
        >>> coord_feat.shape
        torch.Size([4, 2, 32, 32])
    """
    B, C, H, W = feat_shape
    coord = make_coord((H, W), flatten=False).to(device)  # [H, W, 2]
    coord_feat = coord.permute(2, 0, 1).unsqueeze(0).expand(B, 2, H, W)  # [B, 2, H, W]
    return coord_feat


def batched_grid_sample(features, coords, mode='bilinear', align_corners=False):
    """
    批量 Grid Sample 操作的便捷包装

    Args:
        features: [B, C, H, W] - 特征图
        coords: [B, N, 2] - 归一化坐标 [-1, 1]
        mode: 插值模式
        align_corners: 是否对齐角点

    Returns:
        sampled_features: [B, N, C] - 采样的特征

    Example:
        >>> feat = torch.rand(2, 64, 32, 32)
        >>> coords = torch.rand(2, 100, 2) * 2 - 1  # [-1, 1]
        >>> sampled = batched_grid_sample(feat, coords)
        >>> sampled.shape
        torch.Size([2, 100, 64])
    """
    B, C, H, W = features.shape
    # coords 需要翻转 (x, y) -> (y, x) 并添加维度
    sampled = F.grid_sample(
        features,
        coords.flip(-1).unsqueeze(1),  # [B, 1, N, 2]
        mode=mode,
        align_corners=align_corners
    )
    sampled = sampled[:, :, 0, :].permute(0, 2, 1)  # [B, N, C]
    return sampled


def compute_relative_coords(query_coords, feat_coords):
    """
    计算查询坐标相对于特征坐标的相对位置

    Args:
        query_coords: [B, Q, 2] - 查询坐标
        feat_coords: [B, Q, 2] - 最近邻特征坐标

    Returns:
        rel_coords: [B, Q, 2] - 相对坐标

    Example:
        >>> query = torch.tensor([[[0.5, 0.5]]])  # [1, 1, 2]
        >>> feat = torch.tensor([[[0.0, 0.0]]])
        >>> rel = compute_relative_coords(query, feat)
        >>> rel
        tensor([[[0.5, 0.5]]])
    """
    rel_coords = query_coords - feat_coords
    return rel_coords


def unfold_feature(feat, kernel_size, stride):
    """
    使用 Unfold 将特征图分块（用于节省内存）

    Args:
        feat: [B, C, H, W] - 特征图
        kernel_size: (kH, kW) 或 int - 块大小
        stride: (sH, sW) 或 int - 步长

    Returns:
        unfolded: [B, C*kH*kW, L] - 展开的特征
        num_blocks: (num_h, num_w) - 块的数量

    Example:
        >>> feat = torch.rand(2, 64, 32, 32)
        >>> unfolded, (nh, nw) = unfold_feature(feat, kernel_size=8, stride=8)
        >>> unfolded.shape, nh, nw
        (torch.Size([2, 4096, 16]), 4, 4)
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    B, C, H, W = feat.shape

    # 计算块数量
    num_h = (H - kernel_size[0]) // stride[0] + 1
    num_w = (W - kernel_size[1]) // stride[1] + 1

    # Unfold 操作
    unfold = nn.Unfold(kernel_size=kernel_size, stride=stride)
    unfolded = unfold(feat)  # [B, C*kH*kW, L]

    return unfolded, (num_h, num_w)


def fold_feature(unfolded, output_size, kernel_size, stride):
    """
    使用 Fold 将分块特征重组回特征图

    Args:
        unfolded: [B, C*kH*kW, L] - 展开的特征
        output_size: (H, W) - 输出尺寸
        kernel_size: (kH, kW) 或 int - 块大小
        stride: (sH, sW) 或 int - 步长

    Returns:
        folded: [B, C, H, W] - 重组的特征图

    Example:
        >>> unfolded = torch.rand(2, 4096, 16)  # C=64, k=8, L=16
        >>> folded = fold_feature(unfolded, (32, 32), kernel_size=8, stride=8)
        >>> folded.shape
        torch.Size([2, 64, 32, 32])
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    fold = nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride)
    folded = fold(unfolded)  # [B, C, H, W]

    return folded


# ============ 测试代码 ============
if __name__ == '__main__':
    print("========== 测试高斯散射工具函数 ==========\n")

    # 测试 1: make_coord
    print("测试 1: make_coord")
    coord = make_coord((4, 4), flatten=True)
    print(f"  coord.shape: {coord.shape}")
    print(f"  coord 范围: [{coord.min():.3f}, {coord.max():.3f}]")
    print(f"  前 4 个坐标:\n{coord[:4]}\n")

    # 测试 2: make_cell
    print("测试 2: make_cell")
    cell = make_cell(coord, (64, 64))
    print(f"  cell.shape: {cell.shape}")
    print(f"  cell[0]: {cell[0]}\n")

    # 测试 3: to_pixel_samples
    print("测试 3: to_pixel_samples")
    img = torch.rand(2, 3, 8, 8)
    coord, rgb = to_pixel_samples(img)
    print(f"  img.shape: {img.shape}")
    print(f"  coord.shape: {coord.shape}, rgb.shape: {rgb.shape}\n")

    # 测试 4: generate_meshgrid
    print("测试 4: generate_meshgrid")
    coords = generate_meshgrid(3, 3)
    print(f"  coords:\n{coords}\n")

    # 测试 5: fetching_features_from_tensor
    print("测试 5: fetching_features_from_tensor")
    feat = torch.rand(2, 64, 8, 8)
    query_coords = torch.tensor([[0, 0], [3, 3], [7, 7]])
    values, coords_norm = fetching_features_from_tensor(feat, query_coords)
    print(f"  values.shape: {values.shape}")
    print(f"  coords_norm:\n{coords_norm}\n")

    # 测试 6: extract_patch
    print("测试 6: extract_patch")
    img = torch.rand(1, 3, 32, 32)
    patch = extract_patch(img, center=(16, 16), radius=4)
    print(f"  patch.shape: {patch.shape}\n")

    # 测试 7: batched_grid_sample
    print("测试 7: batched_grid_sample")
    feat = torch.rand(2, 64, 16, 16)
    coords = torch.rand(2, 100, 2) * 2 - 1
    sampled = batched_grid_sample(feat, coords)
    print(f"  sampled.shape: {sampled.shape}\n")

    # 测试 8: unfold_feature & fold_feature
    print("测试 8: unfold_feature & fold_feature")
    feat = torch.rand(2, 64, 32, 32)
    unfolded, (nh, nw) = unfold_feature(feat, kernel_size=8, stride=8)
    print(f"  unfolded.shape: {unfolded.shape}, num_blocks: ({nh}, {nw})")
    folded = fold_feature(unfolded, (32, 32), kernel_size=8, stride=8)
    print(f"  folded.shape: {folded.shape}")
    print(f"  重建误差: {(feat - folded).abs().max().item():.6f}\n")

    print("✅ 所有测试通过！")
