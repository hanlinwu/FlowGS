import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ResnetBlock, FRU, StyleLayer_norm_scale_shift
from .gaussian_query_residual_v1 import GaussianSplatterResidualV1


class GAPDecoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        dropout=0.0,
        in_channels,
        z_channels,
        num_sr_modules=[8, 8, 8, 8],
        num_gaussians=100,
        gaussian_kernel_size=5,
        gaussian_hidden_dim=256,
        gaussian_unfold_row=7,
        gaussian_unfold_column=7,
        **ignorekwargs,
    ):
        super().__init__()

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        assert len(num_sr_modules) == len(ch_mult)
        self.style = StyleLayer_norm_scale_shift()

        self.num_sr_modules = num_sr_modules
        self.head = nn.Conv2d(in_channels, ch, 3, padding=1)

        self.FRUList = nn.ModuleList()
        for num in num_sr_modules:
            self.FRUList.append(FRU(ch, num_modules=num))

        self.upsampler = GaussianSplatterResidualV1(
            input_dim=ch,
            num_points=num_gaussians,
            kernel_size=gaussian_kernel_size,
            hidden_dim=gaussian_hidden_dim,
            unfold_row=gaussian_unfold_row,
            unfold_column=gaussian_unfold_column,
        )
        print(
            f"使用高斯残差查询解码器 (num_gaussians={num_gaussians}, kernel_size={gaussian_kernel_size})"
        )

        block_in = ch * ch_mult[-1]
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.transfer = nn.Conv2d(block_in, self.ch, 1)
            up.condition_scale1 = nn.Linear(1, self.ch)
            up.condition_scale2 = nn.Linear(1, self.ch)
            self.up.insert(0, up)

    def get_last_layer(self):
        return self.upsampler.dec.layers[-1].weight

    def forward(self, emb, lr, out_size):
        lr_size = lr.shape[2:]
        scale = torch.tensor([out_size[0] * 1.0 / lr_size[0]], device=lr.device)

        temb = None
        h = self.conv_in(emb)
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)

        h_list, scales1, scales2 = [], [], []
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks):
                h = self.up[i_level].block[i_block](h, temb)
            h_transfer = self.up[i_level].transfer(h)
            h_list.append(F.interpolate(h_transfer, lr_size, mode="nearest"))
            scale1 = self.up[i_level].condition_scale1(scale)
            scale2 = self.up[i_level].condition_scale2(scale)
            scales1.append(scale1.clone())
            scales2.append(scale2.clone())

        head = self.head(lr)
        mid = head
        for i_level in range(self.num_resolutions):
            conditioned = self.style(mid, scales1[i_level], scales2[i_level], h_list[i_level])
            mid = self.FRUList[i_level](conditioned)

        mid = mid + head
        if torch.any(torch.isnan(mid)):
            raise ValueError("NaN detected in SR branch output before residual decoding.")

        residual = self.upsampler(mid, out_size)
        if torch.any(torch.isnan(residual)):
            raise ValueError("NaN detected in residual output.")

        x_up = F.interpolate(lr, out_size, mode="bicubic", align_corners=False)
        out = x_up + residual
        if torch.any(torch.isnan(out)):
            raise ValueError("NaN detected in final reconstruction.")

        return out
