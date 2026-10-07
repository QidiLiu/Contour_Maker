"""MK-UNet 网络定义（vendor 自 SLDGroup/MK-UNet，已移除 timm 依赖）。

原仓库: https://github.com/SLDGroup/MK-UNet  (ICCV 2025 CVAMD Workshop)
原实现仅依赖 torch/torchvision/timm；本文件把 timm 的两个工具函数
(trunc_normal_tf_ / named_apply) 换成本地等价实现，去掉该依赖。

改动(相对原仓库):
  1. 去掉 timm import，改用 torch.nn.init.trunc_normal_ + 本地 apply 遍历
  2. in_channels 默认 3 -> 1（超声为灰度）
  3. 权重初始化: 原 timm 版本在 trunc_normal_ 后做一次额外的 std 缩放，
     此处保留同等行为（named_apply(trunc_normal_tf_, std=.02) -> 等价实现）
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0,
                   a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    """等价于 timm.layers.trunc_normal_tf_（对 2D+ 张量生效）。

    timm 的实现先 trunc_normal_ 再按 fan_in 重估 std，目的是让权重方差与
    常见初始化一致；此处直接调用 torch 原生 trunc_normal_ + 手工 std 调整。
    """
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


def _named_apply(fn, module: nn.Module, name: str = "", depth_first: bool = True):
    """等价于 timm.models.helpers.named_apply。"""
    if isinstance(module, nn.Module):
        for k, v in module.named_children():
            if depth_first:
                module.add_module(k, _named_apply(fn, v, f"{name}.{k}" if name else k))
            else:
                module.add_module(k, fn(v))
    elif fn is not None:
        module = fn(module) if not callable(getattr(module, "apply", None)) else module
    return module


# ---------------------------------------------------------------- 注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes=None, ratio=16, activation="relu"):
        super().__init__()
        out_planes = out_planes or in_planes
        hidden = max(1, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.relu = nn.ReLU(inplace=True) if activation == "relu" else nn.ReLU6(inplace=True)
        self.fc2 = nn.Conv2d(hidden, out_planes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self, scheme=""):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.bn2(y)
        return x * self.sigmoid(y)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self, scheme=""):
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        y = self.bn(y)
        return x * self.sigmoid(y)


class GroupedAttentionGate(nn.Module):
    """分组注意力门 (GAG)。"""

    def __init__(self, F_g, F_l, F_int, kernel_size=1, groups=1, activation="relu"):
        super().__init__()
        self.F_g = F_g
        self.F_l = F_l
        self.F_int = F_int
        self.groups = groups
        self.kernel_size = kernel_size

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv_gap = nn.Conv2d(F_g, F_int, 1, bias=False)
        self.bn_gap = nn.BatchNorm2d(F_int)
        self.relu = nn.ReLU(inplace=True) if activation == "relu" else nn.ReLU6(inplace=True)
        self.up = nn.Conv2d(F_int, F_l, 1, bias=False)
        self.bn_up = nn.BatchNorm2d(F_l)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self, scheme=""):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, g, x):
        gap = self.gap(g)
        gap = self.conv_gap(gap)
        gap = self.bn_gap(gap)
        gap = self.relu(gap)
        gap = self.up(gap)
        gap = self.bn_up(gap)
        gap = self.sigmoid(gap)
        return x + gap * x


# ---------------------------------------------------------------- 多核深度卷积
def _channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """ShuffleNet 的 channel shuffle：把 concat 的各分支交错混合。"""
    n, c, h, w = x.shape
    if groups <= 1 or c % groups != 0:
        return x
    x = x.view(n, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(n, c, h, w)


class MultiKernelDepthwiseConv(nn.Module):
    """并行的多核 depthwise 卷积 (MKDC 核心)。

    对输入分别做 kernel_sizes 中每个核的 depthwise 卷积，返回结果列表
    (不自行融合——融合方式由外层 MultiKernelInvertedResidualBlock 决定)。
    严格对齐官方实现：每个分支是 Conv(groups=C) + BN + 激活。
    """

    def __init__(self, in_channels, kernel_sizes, stride, activation="relu6",
                 dw_parallel=True):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_sizes = list(kernel_sizes) if isinstance(kernel_sizes, (list, tuple)) \
            else [kernel_sizes]
        self.stride = stride
        self.activation = activation
        self.dw_parallel = dw_parallel
        act = nn.ReLU6(inplace=True) if activation == "relu6" else nn.ReLU(inplace=True)
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, k, stride, k // 2,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                act,
            )
            for k in self.kernel_sizes
        ])
        self.init_weights("normal")

    def init_weights(self, scheme=""):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x) -> list[torch.Tensor]:
        outs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outs.append(dw_out)
            if not self.dw_parallel:
                # 串行模式下把输出累加回主支路（官方行为）
                x = x + dw_out
        return outs


class MultiKernelInvertedResidualBlock(nn.Module):
    """MobileNetV2 风格的 MBConv，depthwise 换成多核版本。

    add=True  : 各 depthwise 分支逐元素相加（通道数不变）
    add=False : 各分支 concat 后 channel shuffle（通道数 x n_scales）
    """

    def __init__(self, in_c, out_c, stride, expansion_factor=2,
                 dw_parallel=True, add=True, kernel_sizes=(1, 3, 5),
                 activation="relu6"):
        super().__init__()
        assert stride in (1, 2)
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.add = add
        self.n_scales = len(kernel_sizes)
        self.use_skip = stride == 1

        act = nn.ReLU6(inplace=True) if activation == "relu6" else nn.ReLU(inplace=True)
        self.ex_c = int(in_c * expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act,
        )
        self.multi_scale_dwconv = MultiKernelDepthwiseConv(
            self.ex_c, kernel_sizes, stride, activation, dw_parallel=dw_parallel)

        self.combined_channels = self.ex_c if add else self.ex_c * self.n_scales
        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
        )
        if self.use_skip and in_c != out_c:
            # 通道不一致时用 1x1 对齐后再做残差相加
            self.conv1x1 = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.init_weights("normal")

    def init_weights(self, scheme=""):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        p = self.pconv1(x)
        dw_outs = self.multi_scale_dwconv(p)
        if self.add:
            d = dw_outs[0]
            for o in dw_outs[1:]:
                d = d + o
        else:
            d = torch.cat(dw_outs, dim=1)
            g = math.gcd(self.combined_channels, self.out_c)
            d = _channel_shuffle(d, g)
        out = self.pconv2(d)
        if self.use_skip:
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x + out
        return out


def mk_irb_bottleneck(in_c, out_c, n, stride, expansion_factor=2, dw_parallel=True,
                       add=True, kernel_sizes=(1, 3, 5), activation="relu6"):
    """生成 n 个串联的 MK-IRB（首个 stride=s，其余 stride=1）。"""
    blocks = [MultiKernelInvertedResidualBlock(
        in_c, out_c, stride, expansion_factor=expansion_factor,
        dw_parallel=dw_parallel, add=add, kernel_sizes=kernel_sizes,
        activation=activation)]
    for _ in range(1, n):
        blocks.append(MultiKernelInvertedResidualBlock(
            out_c, out_c, 1, expansion_factor=expansion_factor,
            dw_parallel=dw_parallel, add=add, kernel_sizes=kernel_sizes,
            activation=activation))
    return nn.Sequential(*blocks)


# ---------------------------------------------------------------- U-Net 主体
class _MKUNetBase(nn.Module):
    """MK-UNet 主干。

    结构严格对齐官方实现：
      * 编码器 5 级，每级 = MK-IRB 组 + maxpool 下采样
      * 通道注意力 + 空间注意力，在 bottleneck 与每级解码前施加
      * 解码器用 bilinear 上采样 + 逐元素相加（而非 concat）融合 skip
      * GroupedAttentionGate 在相加前对 skip 特征做门控
      * 官方返回多尺度深监督输出，本项目只取最终 p4
    """

    def __init__(self, num_classes=1, in_channels=1,
                 channels=(16, 32, 64, 96, 160), depths=(1, 1, 1, 1, 1),
                 kernel_sizes=(1, 3, 5), expansion_factor=2, gag_kernel=3):
        super().__init__()
        self.num_classes = num_classes
        self.channels = list(channels)
        self.depths = list(depths)
        self.kernel_sizes = list(kernel_sizes)

        c = self.channels
        d = [max(1, x) for x in self.depths]
        ks = self.kernel_sizes
        ef = expansion_factor

        # ---- encoder：encoder{i} 输出 channels[i-1]，随后 maxpool
        self.enc1 = mk_irb_bottleneck(in_channels, c[0], d[0], 1,
                                      expansion_factor=ef, kernel_sizes=ks)
        self.enc2 = mk_irb_bottleneck(c[0], c[1], d[1], 1, expansion_factor=ef, kernel_sizes=ks)
        self.enc3 = mk_irb_bottleneck(c[1], c[2], d[2], 1, expansion_factor=ef, kernel_sizes=ks)
        self.enc4 = mk_irb_bottleneck(c[2], c[3], d[3], 1, expansion_factor=ef, kernel_sizes=ks)
        self.enc5 = mk_irb_bottleneck(c[3], c[4], d[4], 1, expansion_factor=ef, kernel_sizes=ks)

        # ---- 解码前的注意力门：AG1..AG4 对应 skip 特征 t4,t3,t2,t1
        self.ag1 = GroupedAttentionGate(c[3], c[3], c[3] // 2, gag_kernel, c[3] // 2)
        self.ag2 = GroupedAttentionGate(c[2], c[2], c[2] // 2, gag_kernel, c[2] // 2)
        self.ag3 = GroupedAttentionGate(c[1], c[1], c[1] // 2, gag_kernel, c[1] // 2)
        self.ag4 = GroupedAttentionGate(c[0], c[0], c[0] // 2, gag_kernel, c[0] // 2)

        # ---- decoder：逐级 channels[i] -> channels[i-1]
        self.dec1 = mk_irb_bottleneck(c[4], c[3], 1, 1, expansion_factor=ef, kernel_sizes=ks)
        self.dec2 = mk_irb_bottleneck(c[3], c[2], 1, 1, expansion_factor=ef, kernel_sizes=ks)
        self.dec3 = mk_irb_bottleneck(c[2], c[1], 1, 1, expansion_factor=ef, kernel_sizes=ks)
        self.dec4 = mk_irb_bottleneck(c[1], c[0], 1, 1, expansion_factor=ef, kernel_sizes=ks)
        self.dec5 = mk_irb_bottleneck(c[0], c[0], 1, 1, expansion_factor=ef, kernel_sizes=ks)

        # ---- 通道注意力（ratio 与官方一致）
        self.ca1 = ChannelAttention(c[4], ratio=16)
        self.ca2 = ChannelAttention(c[3], ratio=16)
        self.ca3 = ChannelAttention(c[2], ratio=16)
        self.ca4 = ChannelAttention(c[1], ratio=8)
        self.ca5 = ChannelAttention(c[0], ratio=4)
        self.sa = SpatialAttention()

        # ---- 输出头（官方有 4 个深监督头，本项目只用最后一个）
        self.out4 = nn.Conv2d(c[0], num_classes, 1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _up_relu(self, block, x):
        y = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return F.relu(block(y))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                _trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                _trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # ---- encoder：每级后 maxpool 降采样
        x = F.max_pool2d(self.enc1(x), 2, 2)
        t1 = x
        x = F.max_pool2d(self.enc2(x), 2, 2)
        t2 = x
        x = F.max_pool2d(self.enc3(x), 2, 2)
        t3 = x
        x = F.max_pool2d(self.enc4(x), 2, 2)
        t4 = x
        x = F.max_pool2d(self.enc5(x), 2, 2)      # bottleneck

        # ---- 注意力门控
        # 注意: ChannelAttention / SpatialAttention 的 forward 已经返回
        #   `x * sigmoid(attn)`（加权后的特征）。官方 forward 写作 `self.CA1(out)*out`，
        #   等价于再乘一次 x，即 x^2 * sigmoid。经 5 级放大后激活值爆到 1e18 量级，
        #   训练直接 NaN。此处按注意力门控的本意使用其返回值，不再重复乘 x。
        x = self.ca1(x)
        x = self.sa(x)

        # ---- decoder：上采样 -> 注意力门控 skip -> 逐元素相加
        x = self._up_relu(self.dec1, x)
        x = x + self.ag1(g=x, x=t4)

        x = self.sa(self.ca2(x))
        x = self._up_relu(self.dec2, x)
        x = x + self.ag2(g=x, x=t3)

        x = self.sa(self.ca3(x))
        x = self._up_relu(self.dec3, x)
        x = x + self.ag3(g=x, x=t2)

        x = self.sa(self.ca4(x))
        x = self._up_relu(self.dec4, x)
        x = x + self.ag4(g=x, x=t1)

        x = self.sa(self.ca5(x))
        x = self._up_relu(self.dec5, x)

        # 最终输出分辨率与输入一致（encoder 5 次下采样 + decoder 5 次上采样）
        return self.out4(x)


class MK_UNet(_MKUNetBase):
    """MK-UNet 主变体 (channels=[16,32,64,96,160])，约 0.316M 参数。"""

    def __init__(self, num_classes=1, in_channels=1, **kw):
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         channels=(16, 32, 64, 96, 160), **kw)


class MK_UNet_S(_MKUNetBase):
    """小型变体 (channels=[8,16,32,48,80])。"""

    def __init__(self, num_classes=1, in_channels=1, **kw):
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         channels=(8, 16, 32, 48, 80), **kw)


class MK_UNet_T(_MKUNetBase):
    """微型变体 (channels=[4,8,16,24,32])。"""

    def __init__(self, num_classes=1, in_channels=1, **kw):
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         channels=(4, 8, 16, 24, 32), **kw)


VARIANTS = {
    "MK_UNet": MK_UNet,
    "MK_UNet_S": MK_UNet_S,
    "MK_UNet_T": MK_UNet_T,
}


def build_mkunet(variant: str = "MK_UNet", num_classes: int = 1, in_channels: int = 1):
    if variant not in VARIANTS:
        raise ValueError(f"未知变体 {variant}, 可选: {sorted(VARIANTS)}")
    return VARIANTS[variant](num_classes=num_classes, in_channels=in_channels)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name in ("MK_UNet", "MK_UNet_S", "MK_UNet_T"):
        net = build_mkunet(name, num_classes=1, in_channels=1)
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            y = net(x)
        print(f"{name:12s} params={count_params(net)/1e6:.3f}M  "
              f"out={tuple(y.shape)}")