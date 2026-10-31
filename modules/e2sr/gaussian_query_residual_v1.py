import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaussian_utils import (
    make_coord,
    make_cell,
    generate_meshgrid,
    fetching_features_from_tensor,
)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, act="gelu"):
        super().__init__()
        if act is None:
            self.act = None
        elif act.lower() == "relu":
            self.act = nn.ReLU()
        elif act.lower() == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"activation {act} is not supported")

        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            if self.act:
                layers.append(self.act)
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.contiguous().view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class ClassifierHead(nn.Module):
    def __init__(self, n_feats, n_class=100):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=n_feats, out_channels=n_class, kernel_size=3, padding=1
        )

    def forward(self, x):
        logits = self.conv2d(x)
        batch, n_class, height, width = logits.shape
        logits = logits.permute(0, 2, 3, 1).contiguous().view(batch * height * width, n_class)
        if self.training:
            logits = F.gumbel_softmax(logits, tau=1, hard=False)
        else:
            logits = F.gumbel_softmax(logits, tau=1, hard=True)
        logits = logits.view(batch, height, width, n_class).permute(0, 3, 1, 2).contiguous()
        return logits


class GaussianSplatterResidualV1(nn.Module):
    def __init__(
        self,
        input_dim,
        kernel_size,
        hidden_dim=256,
        unfold_row=7,
        unfold_column=7,
        num_points=100,
    ):
        super().__init__()

        self.classifier_head = ClassifierHead(n_feats=input_dim, n_class=num_points)

        self.feat = None
        self.logits = None
        self.feat_coord = None

        self.coef = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.phase = nn.Linear(2, hidden_dim // 2, bias=False)
        self.dec = MLP(in_dim=hidden_dim, out_dim=3, hidden_list=[256, 256, 256, 256])

        self.kernel_size = kernel_size
        self.row = unfold_row
        self.column = unfold_column
        self.num_points = num_points

        sigma_x, sigma_y = torch.meshgrid(
            torch.linspace(0.2, 3.0, 10),
            torch.linspace(0.2, 3.0, 10),
            indexing="ij",
        )
        self.sigma_x = nn.Parameter(sigma_x.reshape(-1))
        self.sigma_y = nn.Parameter(sigma_y.reshape(-1))
        self.opacity = nn.Parameter(torch.sigmoid(torch.ones(self.num_points, 1)))
        self.rho = nn.Parameter(torch.zeros(self.num_points, 1))

    def weighted_gaussian_parameters(self, logits):
        patch_batch, _, _, _ = logits.size()
        logits = logits.permute(0, 2, 3, 1)

        sigma_x = (logits * self.sigma_x.view(1, 1, 1, -1)).sum(dim=-1)
        sigma_y = (logits * self.sigma_y.view(1, 1, 1, -1)).sum(dim=-1)
        opacity = (logits * self.opacity[:, 0].view(1, 1, 1, -1)).sum(dim=-1)
        rho = (logits * self.rho[:, 0].view(1, 1, 1, -1)).sum(dim=-1)

        return (
            sigma_x.reshape(patch_batch, -1),
            sigma_y.reshape(patch_batch, -1),
            opacity.reshape(patch_batch, -1),
            rho.reshape(patch_batch, -1),
        )

    def gen_feat(self, feat):
        self.feat = feat
        self.logits = self.classifier_head(feat)
        self.feat_coord = (
            make_coord(feat.shape[-2:], flatten=False)
            .to(feat.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(feat.shape[0], 2, *feat.shape[-2:])
        )
        return self.feat, self.logits

    def query_rgb(self, coord, scale, cell=None):
        feat, lr_feat, logits = self.feat[:, :8, :, :], self.feat[:, 8:, :, :], self.logits
        feat_size, feat_device = feat.shape, feat.device

        scale = float(scale[0])
        hr_h = round(feat.shape[-2] * scale)
        hr_w = round(feat.shape[-1] * scale)

        num_kernels_row = math.ceil(feat_size[-2] / self.row)
        num_kernels_column = math.ceil(feat_size[-1] / self.column)
        upsampled_size = (num_kernels_row * self.row, num_kernels_column * self.column)
        upsampled_feat = F.interpolate(feat, size=upsampled_size, mode="bicubic", align_corners=False)
        upsampled_logits = F.interpolate(logits, size=upsampled_size, mode="bicubic", align_corners=False)

        unfold = nn.Unfold(kernel_size=(self.row, self.column), stride=(self.row, self.column))
        unfolded_feature = unfold(upsampled_feat)
        unfolded_logits = unfold(upsampled_logits)

        num_patches = unfolded_feature.shape[-1]
        patch_batch = feat_size[0] * num_patches
        unfold_feat = (
            unfolded_feature.transpose(1, 2)
            .reshape(patch_batch, feat_size[1], self.row, self.column)
            .contiguous()
        )
        unfold_logits = (
            unfolded_logits.transpose(1, 2)
            .reshape(patch_batch, logits.shape[1], self.row, self.column)
            .contiguous()
        )

        coords_ = generate_meshgrid(unfold_feat.shape[-2], unfold_feat.shape[-1], device=feat_device)
        num_lr_points = unfold_feat.shape[-2] * unfold_feat.shape[-1]
        colors_, coords_norm = fetching_features_from_tensor(unfold_feat, coords_)

        sigma_x, sigma_y, opacity, rho = self.weighted_gaussian_parameters(unfold_logits)
        rho = rho.clamp(min=-0.999, max=0.999)

        covariance = torch.stack(
            (
                torch.stack((sigma_x.square() + 1e-5, rho * sigma_x * sigma_y), dim=-1),
                torch.stack((rho * sigma_x * sigma_y, sigma_y.square() + 1e-5), dim=-1),
            ),
            dim=-2,
        )
        inv_covariance = torch.inverse(covariance).to(feat_device)

        axis = torch.linspace(-5.0, 5.0, steps=self.kernel_size, device=feat_device)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        xy = torch.stack((xx, yy), dim=-1)

        z = torch.einsum("xyi,bpij,xyj->bpxy", xy, -0.5 * inv_covariance, xy)
        det_covariance = torch.det(covariance).clamp_min(1e-8)
        kernel = torch.exp(z) / (
            2 * torch.tensor(math.pi, device=feat_device) * torch.sqrt(det_covariance)[..., None, None]
        )
        kernel = kernel / kernel.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-8)

        channel = unfold_feat.shape[1]
        kernel_color = (
            kernel.unsqueeze(2)
            .expand(patch_batch, num_lr_points, channel, self.kernel_size, self.kernel_size)
            .contiguous()
            .view(patch_batch * num_lr_points, channel, self.kernel_size, self.kernel_size)
        )

        pad_h = round(unfold_feat.shape[-2] * scale) - self.kernel_size
        pad_w = round(unfold_feat.shape[-1] * scale) - self.kernel_size
        if pad_h < 0 or pad_w < 0:
            raise ValueError("Kernel size should be smaller or equal to the image size.")
        padding = (pad_w // 2, pad_w // 2 + pad_w % 2, pad_h // 2, pad_h // 2 + pad_h % 2)
        kernel_color_padded = F.pad(kernel_color, padding, "constant", 0)

        _, _, kernel_h_full, kernel_w_full = kernel_color_padded.shape
        kernel_color_padded = kernel_color_padded.view(
            patch_batch, num_lr_points, channel, kernel_h_full, kernel_w_full
        )

        theta = torch.zeros(patch_batch, num_lr_points, 2, 3, dtype=torch.float32, device=feat_device)
        theta[:, :, 0, 0] = 1.0
        theta[:, :, 1, 1] = 1.0
        theta[:, :, :, 2] = coords_norm.unsqueeze(0)

        grid = F.affine_grid(
            theta.view(-1, 2, 3),
            size=[patch_batch * num_lr_points, channel, kernel_h_full, kernel_w_full],
            align_corners=True,
        ).contiguous()
        kernel_color_padded = kernel_color_padded.view(
            patch_batch * num_lr_points, channel, kernel_h_full, kernel_w_full
        )
        with torch.backends.cudnn.flags(enabled=False):
            kernel_color_translated = F.grid_sample(
                kernel_color_padded.contiguous(),
                grid.contiguous(),
                align_corners=True,
            )
        kernel_color_translated = kernel_color_translated.view(
            patch_batch, num_lr_points, channel, kernel_h_full, kernel_w_full
        )

        colors = colors_ * opacity.unsqueeze(-1)
        final_image_layers = colors.unsqueeze(-1).unsqueeze(-1) * kernel_color_translated
        final_image = final_image_layers.sum(dim=1)

        kernel_h = round(self.row * scale)
        kernel_w = round(self.column * scale)
        fold = nn.Fold(
            output_size=(kernel_h * num_kernels_row, kernel_w * num_kernels_column),
            kernel_size=(kernel_h, kernel_w),
            stride=(kernel_h, kernel_w),
        )
        final_image = final_image.reshape(
            feat_size[0], num_patches, feat_size[1] * kernel_h * kernel_w
        ).transpose(1, 2)
        final_image = fold(final_image)
        final_image = F.interpolate(final_image, size=(hr_h, hr_w), mode="bicubic", align_corners=False)

        lr_feat = F.interpolate(lr_feat, size=(hr_h, hr_w), mode="bicubic", align_corners=False)
        final_image = torch.cat((final_image, lr_feat), dim=1)

        coef = self.coef(final_image)
        freq = self.freq(final_image)
        feat_coord = self.feat_coord
        coord_ = coord.clone()
        q_coef = (
            F.grid_sample(coef, coord_.flip(-1).unsqueeze(1), mode="nearest", align_corners=False)[:, :, 0, :]
            .permute(0, 2, 1)
            .contiguous()
        )
        q_freq = (
            F.grid_sample(freq, coord_.flip(-1).unsqueeze(1), mode="nearest", align_corners=False)[:, :, 0, :]
            .permute(0, 2, 1)
            .contiguous()
        )
        q_coord = (
            F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode="nearest", align_corners=False)[:, :, 0, :]
            .permute(0, 2, 1)
            .contiguous()
        )

        rel_coord = coord - q_coord
        rel_coord[:, :, 0] *= feat.shape[-2]
        rel_coord[:, :, 1] *= feat.shape[-1]
        rel_cell = cell.clone()
        rel_cell[:, :, 0] *= feat.shape[-2]
        rel_cell[:, :, 1] *= feat.shape[-1]

        batch, query = coord.shape[:2]
        q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
        q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
        q_freq = torch.sum(q_freq, dim=-2)
        q_freq += self.phase(rel_cell.view(batch * query, -1)).view(batch, query, -1)
        q_freq = torch.cat((torch.cos(math.pi * q_freq), torch.sin(math.pi * q_freq)), dim=-1)

        inp = torch.mul(q_coef, q_freq)
        pred = self.dec(inp.contiguous().view(batch * query, -1)).view(batch, query, -1)
        return pred

    def forward(self, feat, out_size):
        self.gen_feat(feat)

        batch = feat.shape[0]
        scale = torch.tensor([out_size[0] * 1.0 / feat.shape[2]], device=feat.device)

        coord = make_coord(out_size, flatten=True).to(feat.device)
        cell = make_cell(coord, out_size)

        coord = coord.unsqueeze(0).expand(batch, -1, -1)
        cell = cell.unsqueeze(0).expand(batch, -1, -1)

        residual = self.query_rgb(coord, scale, cell)
        return residual.view(batch, out_size[0], out_size[1], 3).permute(0, 3, 1, 2).contiguous()
