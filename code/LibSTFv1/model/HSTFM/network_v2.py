import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class ConvAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.GroupNorm(8 if out_channels >= 8 else 1, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.body(x)


class DepthwiseResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            ConvAct(channels, channels, groups=channels),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(8 if channels >= 8 else 1, channels),
        )

    def forward(self, x):
        return x + self.body(x)


class ResidualStack(nn.Module):
    def __init__(self, channels, depth):
        super().__init__()
        self.blocks = nn.Sequential(*[DepthwiseResBlock(channels) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(x)


class S2StructureEncoder(nn.Module):
    """High-resolution S2 structure pyramid used by alignment and decoding."""

    def __init__(self, in_channels, channels):
        super().__init__()
        self.stem = nn.Sequential(ConvAct(in_channels, channels), ResidualStack(channels, 2))
        self.down1 = nn.Sequential(ConvAct(channels, channels, stride=2), ResidualStack(channels, 1))
        self.down2 = nn.Sequential(ConvAct(channels, channels, stride=2), ResidualStack(channels, 1))
        self.down3 = nn.Sequential(ConvAct(channels, channels, stride=2), ResidualStack(channels, 1))

    def forward(self, x):
        f0 = self.stem(x)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        return f0, f1, f2, f3

class MultiScaleStructureProjector(nn.Module):
    """Condense HR S2 structure into the LR alignment grid."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.project = nn.Sequential(
            ConvAct(in_channels * 4, out_channels),
            ResidualStack(out_channels, 2),
        )

    def forward(self, pyramid):
        f0, f1, f2, f3 = pyramid
        size = f3.shape[-2:]
        structure = torch.cat(
            [
                F.adaptive_avg_pool2d(f0, size),
                F.adaptive_avg_pool2d(f1, size),
                F.adaptive_avg_pool2d(f2, size),
                f3,
            ],
            dim=1,
        )
        return self.project(structure)

class LowResEncoder(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.body = nn.Sequential(
            ConvAct(in_channels, channels),
            ResidualStack(channels, 2),
        )

    def forward(self, x):
        return self.body(x)


class DynamicSensorTransferBlock(nn.Module):
    """Low-resolution dynamic S3 mixing plus non-linear spectral transfer."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel_area = kernel_size * kernel_size
        self.spatial_mix = nn.Sequential(
            ConvAct(channels * 3 + 1, channels),
            ResidualStack(channels, 1),
        )
        self.kernel_head = nn.Conv2d(channels, self.kernel_area, 1)
        self.spectral_mix = nn.Sequential(
            ConvAct(channels * 3 + 1, channels),
            ResidualStack(channels, 2),
        )
        self.spectral_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, s3_t1, s3_t2, s2_ref, mask):
        context = self.spatial_mix(torch.cat([s3_t1, s3_t2, s2_ref, mask], dim=1))
        weights = torch.softmax(self.kernel_head(context), dim=1)

        b, c, h, w = s3_t2.shape
        patches = F.unfold(s3_t2, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        patches = patches.view(b, c, self.kernel_area, h, w)
        dynamic_s3_t2 = (patches * weights.unsqueeze(1)).sum(dim=2)

        x = self.spectral_mix(torch.cat([s3_t1, dynamic_s3_t2, s2_ref, mask], dim=1))
        transfer = x * self.spectral_gate(x)
        return transfer, weights


class OffsetHead(nn.Module):
    """Predict one deformable alignment field and modulation mask."""

    def __init__(self, channels, kernel_size=5, deform_groups=8):
        super().__init__()
        self.kernel_size = kernel_size
        self.deform_groups = deform_groups
        self.context = nn.Sequential(
            ConvAct(channels * 4 + 1, channels),
            DepthwiseResBlock(channels),
            ConvAct(channels, channels, kernel_size=1),
        )
        out_channels = deform_groups * 3 * kernel_size * kernel_size
        self.head = nn.Conv2d(channels, out_channels, 1)
        self.deform = DeformConv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=deform_groups,
        )

    def forward(self, source, s2_ref, s3_context, transfer_context, mask):
        context = self.context(torch.cat([source, s2_ref, s3_context, transfer_context, mask], dim=1))
        offset_1, offset_2, modulator = torch.chunk(self.head(context), 3, dim=1)
        offset = torch.cat([offset_1, offset_2], dim=1)
        modulator = torch.sigmoid(modulator)
        aligned = self.deform(source, offset, modulator)
        return aligned, context, offset, modulator


class DualEventAlignment(nn.Module):
    """Separate t1/t2 alignment heads; t2 is event-aware instead of reusing t1 offsets."""

    def __init__(self, channels):
        super().__init__()
        self.transfer = DynamicSensorTransferBlock(channels)
        self.offset_t1 = OffsetHead(channels)
        self.offset_t2 = OffsetHead(channels)
        self.fuse = nn.Sequential(
            ConvAct(channels * 4 + 1, channels),
            ResidualStack(channels, 2),
        )

    def forward(self, s3_t1, s3_t2, s2_ref, mask):
        transfer, transfer_weights = self.transfer(s3_t1, s3_t2, s2_ref, mask)
        aligned_t1, ctx_t1, offset_t1, mod_t1 = self.offset_t1(
            s3_t1, s2_ref, s3_t2, transfer, torch.ones_like(mask)
        )
        aligned_t2, ctx_t2, offset_t2, mod_t2 = self.offset_t2(
            s3_t2, s2_ref, s3_t1, transfer, mask
        )
        fused = self.fuse(torch.cat([aligned_t1, aligned_t2, transfer, s2_ref, mask], dim=1))
        aux = {
            "align_context_t1": ctx_t1,
            "align_context_t2": ctx_t2,
            "offset_t1": offset_t1,
            "offset_t2": offset_t2,
            "modulator_t1": mod_t1,
            "modulator_t2": mod_t2,
            "transfer": transfer,
            "transfer_weights": transfer_weights,
        }
        return fused, aux


class MaskAwareGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            ConvAct(channels * 2 + 2, channels),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, current, skip, mask, change):
        gate = self.gate(torch.cat([current, skip, mask, change], dim=1))
        return current * (1.0 - gate) + skip * gate


class TextureRouter(nn.Module):
    """Route between S2 texture migration and generated texture residual."""

    def __init__(self, channels, out_channels):
        super().__init__()
        self.keep_gate = nn.Sequential(
            ConvAct(channels * 2 + 2, channels),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )
        self.generated_texture = nn.Sequential(
            ConvAct(channels * 2 + 2, channels),
            ResidualStack(channels, 2),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )
        self.s2_texture = nn.Sequential(
            ConvAct(channels, channels),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

    def forward(self, decoded, s2_detail, mask, change):
        route_input = torch.cat([decoded, s2_detail, mask, change], dim=1)
        keep = self.keep_gate(route_input)
        migrated = self.s2_texture(s2_detail)
        generated = self.generated_texture(route_input)
        return migrated * keep + generated * (1.0 - keep), keep


class ReconstructionDecoder(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.up2 = nn.Sequential(ConvAct(channels * 2, channels * 4), nn.PixelShuffle(2))
        self.gate2 = MaskAwareGate(channels)
        self.refine2 = ResidualStack(channels, 2)

        self.up1 = nn.Sequential(ConvAct(channels, channels * 4), nn.PixelShuffle(2))
        self.gate1 = MaskAwareGate(channels)
        self.refine1 = ResidualStack(channels, 2)

        self.up0 = nn.Sequential(ConvAct(channels, channels * 4), nn.PixelShuffle(2))
        self.gate0 = MaskAwareGate(channels)
        self.refine0 = ResidualStack(channels, 3)

        self.low_residual = nn.Conv2d(channels, out_channels, 3, padding=1)
        self.texture = TextureRouter(channels, out_channels)

    def forward(self, fused_low, s2_pyramid, mask, change):
        s2_0, s2_1, s2_2, s2_3 = s2_pyramid

        x = self.up2(torch.cat([fused_low, s2_3], dim=1))
        x = self.gate2(x, s2_2, F.interpolate(mask, size=s2_2.shape[-2:], mode="nearest"),
                       F.interpolate(change, size=s2_2.shape[-2:], mode="bilinear", align_corners=False))
        x = self.refine2(x)

        x = self.up1(x)
        x = self.gate1(x, s2_1, F.interpolate(mask, size=s2_1.shape[-2:], mode="nearest"),
                       F.interpolate(change, size=s2_1.shape[-2:], mode="bilinear", align_corners=False))
        x = self.refine1(x)

        x = self.up0(x)
        x = self.gate0(x, s2_0, F.interpolate(mask, size=s2_0.shape[-2:], mode="nearest"),
                       F.interpolate(change, size=s2_0.shape[-2:], mode="bilinear", align_corners=False))
        x = self.refine0(x)

        base_residual = self.low_residual(x)
        texture_residual, texture_keep = self.texture(
            x,
            s2_0,
            F.interpolate(mask, size=s2_0.shape[-2:], mode="nearest"),
            F.interpolate(change, size=s2_0.shape[-2:], mode="bilinear", align_corners=False),
        )
        return base_residual + texture_residual, {"texture_keep": texture_keep}


class HSTFMv2(nn.Module):
    """A cleaner event-aware HSTFM variant.

    Inputs follow the original raw network order:
        a: S3/LR at t1 already enlarged to HR grid, [B, C, H, W]
        A: S2/HR at t1, [B, C, H, W]
        b: S3/LR at t2 already enlarged to HR grid, [B, C, H, W]
        mask: valid S3_t2 observation mask, [B, 1, H, W]

    Output is a residual prediction around S2_t1:
        pred = A + residual
    """

    def __init__(self, in_channels=4, dim=96, align_dim=64, cf_ratio=8):
        super().__init__()
        self.cf_ratio = int(cf_ratio)
        self.s2_encoder = S2StructureEncoder(in_channels, dim)
        self.s2_to_align = MultiScaleStructureProjector(dim, align_dim)
        self.s3_encoder = LowResEncoder(in_channels, align_dim)
        self.change_encoder = nn.Sequential(
            ConvAct(in_channels + 1, align_dim),
            ResidualStack(align_dim, 1),
        )
        self.alignment = DualEventAlignment(align_dim)
        self.low_fusion = nn.Sequential(
            ConvAct(align_dim * 2, dim),
            ResidualStack(dim, 3),
        )
        self.decoder = ReconstructionDecoder(dim, in_channels)

    def downsample_lr_grid(self, x):
        return F.avg_pool2d(x, kernel_size=self.cf_ratio, stride=self.cf_ratio)

    def masked_downsample_lr_grid(self, x, mask, eps=1e-6):
        valid = self.downsample_lr_grid(mask).clamp(0.0, 1.0)
        summed = F.avg_pool2d(x * mask, kernel_size=self.cf_ratio, stride=self.cf_ratio)
        return summed / valid.clamp_min(eps), valid

    def forward(self, a, A, b, mask, return_aux=False):
        s2_pyramid = self.s2_encoder(A)
        s2_ref_low = self.s2_to_align(s2_pyramid)

        s3_t1_low = self.downsample_lr_grid(a)
        s3_t2_low, mask_low = self.masked_downsample_lr_grid(b, mask)

        s3_t1_feat = self.s3_encoder(s3_t1_low)
        s3_t2_feat = self.s3_encoder(s3_t2_low)
        change_low = torch.mean(torch.abs(s3_t2_low - s3_t1_low), dim=1, keepdim=True)
        change_feat = self.change_encoder(torch.cat([s3_t2_low - s3_t1_low, mask_low], dim=1))

        aligned_low, align_aux = self.alignment(s3_t1_feat, s3_t2_feat, s2_ref_low, mask_low)
        fused_low = self.low_fusion(torch.cat([aligned_low, change_feat], dim=1))
        residual, dec_aux = self.decoder(fused_low, s2_pyramid, mask_low, change_low)
        pred = A + residual

        if return_aux:
            aux = {**align_aux, **dec_aux, "change_low": change_low}
            return pred, aux
        return pred


class HSTFM(HSTFMv2):
    """Compatibility alias for experiments that import HSTFM from this file."""

    pass


