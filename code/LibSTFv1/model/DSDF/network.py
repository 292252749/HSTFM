import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
import warnings

try:
    import triton
    import triton.language as tl
    WITH_TRITON = True
except ImportError:
    WITH_TRITON = False
    warnings.warn("Triton not installed, fall back to pytorch implements.")

from mamba_ssm import Mamba2

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

def cross_scan_fwd(x: torch.Tensor):
    B, C, H, W = x.shape
    y = x.new_empty((B, 4, C, H * W))
    y[:, 0, :, :] = x.flatten(2, 3)
    y[:, 1, :, :] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
    y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])
    return y

def cross_merge_fwd(y: torch.Tensor):
    B, K, D, H, W = y.shape
    y = y.view(B, K, D, -1)
    res = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
    res = res[:, 0] + res[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
    return res.view(B, D, H, W)

if WITH_TRITON:
    @triton.jit
    def triton_cross_scan_flex(
        x: tl.tensor, y: tl.tensor, x_layout: tl.constexpr, y_layout: tl.constexpr, 
        operation: tl.constexpr, onebyone: tl.constexpr, scans: tl.constexpr, 
        BC: tl.constexpr, BH: tl.constexpr, BW: tl.constexpr, DC: tl.constexpr, 
        DH: tl.constexpr, DW: tl.constexpr, NH: tl.constexpr, NW: tl.constexpr,
    ):
        i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_h, i_w = (i_hw // NW), (i_hw % NW)
        _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
        _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
        _mask_hw = _mask_h[:, None] & _mask_w[None, :]
        _for_C = min(DC - i_c * BC, BC)

        pos_h = (i_h * BH + tl.arange(0, BH)[:, None])
        pos_w = (i_w * BW + tl.arange(0, BW)[None, :])
        neg_h = (DH - i_h * BH - 1 - tl.arange(0, BH)[:, None])
        neg_w = (DW - i_w * BW - 1 - tl.arange(0, BW)[None, :])
        
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = pos_w * DH + pos_h
        HWRoute2 = neg_h * DW + neg_w
        HWRoute3 = neg_w * DH + neg_h

        _tmp1 = DC * DH * DW
        y_ptr_base = y + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if y_layout == 0 else i_c * BC)
        p_y1 = y_ptr_base + HWRoute0
        p_y2 = y_ptr_base + _tmp1 + HWRoute1
        p_y3 = y_ptr_base + 2 * _tmp1 + HWRoute2
        p_y4 = y_ptr_base + 3 * _tmp1 + HWRoute3
        
        x_ptr_base = x + i_b * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
        p_x = x_ptr_base + HWRoute0

        if operation == 0:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                _x = tl.load(p_x + _idx_x, mask=_mask_hw)
                tl.store(p_y1 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y2 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y3 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y4 + _idx_y, _x, mask=_mask_hw)
        elif operation == 1:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                _y1 = tl.load(p_y1 + _idx_y, mask=_mask_hw)
                _y2 = tl.load(p_y2 + _idx_y, mask=_mask_hw)
                _y3 = tl.load(p_y3 + _idx_y, mask=_mask_hw)
                _y4 = tl.load(p_y4 + _idx_y, mask=_mask_hw)
                tl.store(p_x + _idx_x, _y1 + _y2 + _y3 + _y4, mask=_mask_hw)

    class CrossScanTritonF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor):
            B, C, H, W = x.shape
            BC, BH, BW = 1, 32, 32
            NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
            ctx.shape = (B, C, H, W)
            ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
            y = x.new_empty((B, 4, C, H * W))
            triton_cross_scan_flex[(NH * NW, NC, B)](
                x.contiguous(), y, 0, 0, 0, 0, 0, BC, BH, BW, C, H, W, NH, NW
            )
            return y
            
        @staticmethod
        def backward(ctx, y: torch.Tensor):
            B, C, H, W = ctx.shape
            BC, BH, BW, NC, NH, NW = ctx.triton_shape
            x = y.new_empty((B, C, H, W))
            triton_cross_scan_flex[(NH * NW, NC, B)](
                x, y.contiguous(), 0, 0, 1, 0, 0, BC, BH, BW, C, H, W, NH, NW
            )
            return x

    class CrossMergeTritonF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y: torch.Tensor):
            B, _, C, H, W = y.shape
            BC, BH, BW = 1, 32, 32
            NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
            ctx.shape = (B, C, H, W)
            ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
            x = y.new_empty((B, C, H * W))
            triton_cross_scan_flex[(NH * NW, NC, B)](
                x, y.contiguous(), 0, 0, 1, 0, 0, BC, BH, BW, C, H, W, NH, NW
            )
            return x.view(B, C, H, W)
            
        @staticmethod
        def backward(ctx, x: torch.Tensor):
            B, C, H, W = ctx.shape
            BC, BH, BW, NC, NH, NW = ctx.triton_shape
            y = x.new_empty((B, 4, C, H, W))
            triton_cross_scan_flex[(NH * NW, NC, B)](
                x.contiguous(), y, 0, 0, 0, 0, 0, BC, BH, BW, C, H, W, NH, NW
            )
            return y

def cross_scan_fn(x):
    if WITH_TRITON and x.is_cuda:
        return CrossScanTritonF.apply(x)
    return cross_scan_fwd(x)

def cross_merge_fn(y):
    if WITH_TRITON and y.is_cuda:
        return CrossMergeTritonF.apply(y)
    return cross_merge_fwd(y)

class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight + self.bias



class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden_dim = dim * mult
        self.net = nn.Sequential(
            LayerNorm2d(dim),
            nn.Conv2d(dim, hidden_dim, 1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )
        
    def forward(self, x):
        return self.net(x)

        
class SS2DMamba(nn.Module):
    def __init__(self, dim, d_state=64, expand=2):
        super().__init__()
        self.dim = dim
        self.mamba = Mamba2(d_model=dim, d_state=d_state, expand=expand, headdim=64) if Mamba2 else nn.Identity()
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        xs = cross_scan_fn(x) 
        xs = xs.transpose(2, 3).reshape(B * 4, H * W, C)
        
        x_out = self.mamba(xs)
        x_out = self.out_norm(x_out)
        
        x_out = x_out.view(B, 4, H * W, C).transpose(2, 3).view(B, 4, C, H, W)
        return cross_merge_fn(x_out)



class GSE(nn.Module):
    def __init__(self, dim, d_state=64, expand=2, drop_path=0.):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.ss2d = SS2DMamba(dim, d_state, expand)
        self.gamma = nn.Parameter(1e-6 * torch.ones(1, dim, 1, 1), requires_grad=True)
        self.ffn = FeedForward(dim)
        self.gamma_ffn = nn.Parameter(1e-6 * torch.ones(1, dim, 1, 1), requires_grad=True)
        self.feature_dropout = nn.Dropout2d(0.4) 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.ss2d(x)
        x = self.feature_dropout(x)
        x = shortcut + self.drop_path(x * self.gamma)
        shortcut = x
        x = self.ffn(x)
        x = self.feature_dropout(x)
        x = shortcut + self.drop_path(x * self.gamma_ffn)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  
        self.num_heads = num_heads
        
        self.register_buffer("relative_position_index", self.get_relative_position_index(window_size))
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        
        self.attn_drop = nn.Dropout(0.)
        self.proj_drop = nn.Dropout(0.)

    def get_relative_position_index(self, window_size):
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  
        coords_flatten = torch.flatten(coords, 1)  
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        return relative_coords.sum(-1)

    def forward(self, q, k, v):
        B_, N, C = q.shape
        q = q.view(B_, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B_, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B_, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        
        attn_mask = relative_position_bias.unsqueeze(0)
        
        x = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj_drop(x)
        return x

class SDA(nn.Module):
    def __init__(self, dim, heads=8, window_size=16):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        
        self.norm_h  = LayerNorm2d(dim)
        self.norm_l1 = LayerNorm2d(dim)
        self.norm_l2 = LayerNorm2d(dim)
        
        self.q_proj_h  = nn.Conv2d(dim, dim, 1, bias=False)
        self.q_proj_l1 = nn.Conv2d(dim, dim, 1, bias=False)
        
        self.k_proj_l1 = nn.Conv2d(dim, dim, 1, bias=False)
        
        self.v_proj_l2 = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_proj_l1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_act     = nn.GELU()
        
        self.attn = WindowAttention(dim, (window_size, window_size), heads)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x_high, x_l1, x_l2):
        shortcut = x_high
        
        x_high = self.norm_h(x_high)
        x_l1   = self.norm_l1(x_l1)
        x_l2   = self.norm_l2(x_l2)
        
        B, C, H, W = x_high.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            pad = (0, pad_w, 0, pad_h)
            x_high = F.pad(x_high, pad)
            x_l1   = F.pad(x_l1, pad)
            x_l2   = F.pad(x_l2, pad)
            
        Hp, Wp = x_high.shape[2], x_high.shape[3]
        
        q = self.q_proj_h(x_high) + self.q_proj_l1(x_high - x_l1)
        k = self.k_proj_l1(x_l1)
        v = self.v_act(self.v_proj_l2(x_l2) + self.v_proj_l1(x_l2 - x_l1))
        
        q_w = rearrange(q, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        k_w = rearrange(k, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        v_w = rearrange(v, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        
        attn_out = self.attn(q_w, k_w, v_w)
        
        x = rearrange(attn_out, '(b h w) (p1 p2) c -> b c (h p1) (w p2)', 
                      h=Hp//self.window_size, w=Wp//self.window_size, 
                      p1=self.window_size, p2=self.window_size)
        
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
            
        return self.proj(x) + shortcut

class LSR(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)
        self.gamma = nn.Parameter(1e-6 * torch.ones(1, dim, 1, 1), requires_grad=True)
        self.spatial_dropout = nn.Dropout2d(0.4)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.spatial_dropout(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return input + self.drop_path(x * self.gamma)

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, layer_type='LSR', drop_path=0., use_checkpoint=False):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.use_checkpoint = use_checkpoint
        
        if isinstance(drop_path, float):
            drop_path = [drop_path] * depth
        
        for i in range(depth):
            if layer_type == 'LSR':
                self.blocks.append(LSR(dim, drop_path=drop_path[i]))
            elif layer_type == 'GSE':
                self.blocks.append(GSE(dim, drop_path=drop_path[i]))
                
    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x

class DSDF(nn.Module):
    def __init__(self, in_channels, inter_dim=64, dim=64, dim_mults=(1, 2, 4), 
                 drop_path_rate=0.2, use_checkpoint=True): 
        super().__init__()
        
        self.high_init = nn.Conv2d(in_channels, inter_dim, 3, 1, 1)
        self.low_init  = nn.Conv2d(in_channels, inter_dim, 3, 1, 1)
        self.sda       = SDA(inter_dim, heads=8, window_size=16)
        
        self.raw_injection = nn.Sequential(
            nn.Conv2d(inter_dim * 2, dim, 1, bias=False),
            LayerNorm2d(dim),
            nn.GELU() 
        )
        
        self.downs = nn.ModuleList([])
        dims = [dim, *[dim * m for m in dim_mults]] 
        in_out = list(zip(dims[:-1], dims[1:]))
        
        enc_depths = [2, 2, 2]
        dec_depths = [2, 2, 2]
        mid_depth = 2
        enc_types  = ['LSR', 'LSR', 'GSE']
        dec_types  = ['GSE', 'LSR', 'LSR']
        
        total_depth = sum(enc_depths) + mid_depth + sum(dec_depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        cur = 0
        
        for i, (dim_in, dim_out) in enumerate(in_out):
            layer = BasicLayer(dim_in, depth=enc_depths[i], layer_type=enc_types[i],
                               drop_path=dpr[cur:cur+enc_depths[i]], use_checkpoint=use_checkpoint)
            cur += enc_depths[i]
            down = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(dim_in*4, dim_out, 1))
            self.downs.append(nn.ModuleList([layer, down]))

        self.mid_block = BasicLayer(dims[-1], depth=mid_depth, layer_type='GSE',
                                    drop_path=dpr[cur:cur+mid_depth], use_checkpoint=use_checkpoint)
                                    
        cur += mid_depth
        
        self.ups = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):
            up = nn.Sequential(nn.Conv2d(dim_out, dim_in*4, 1), nn.PixelShuffle(2))
            reduce = nn.Conv2d(dim_in*2, dim_in, 1)
            layer = BasicLayer(dim_in, depth=dec_depths[i], layer_type=dec_types[i], 
                               drop_path=dpr[cur:cur+dec_depths[i]], use_checkpoint=use_checkpoint)
            cur += dec_depths[i]
            self.ups.append(nn.ModuleList([up, reduce, layer]))
            
        self.final_conv = nn.Conv2d(dims[0], in_channels, 1)

    def forward(self, H1, L1, L2):
        L1 = torch.clamp(L1, 0, 1)
        L2 = torch.clamp(L2, 0, 1)
        H1 = torch.clamp(H1, 0, 1)
        h_feat  = self.high_init(H1)
        l1_feat = self.low_init(L1)
        l2_feat = self.low_init(L2)
        
        aligned_feat = self.sda(h_feat, l1_feat, l2_feat)
        
        x_in = torch.cat([aligned_feat, h_feat], dim=1)
        x = self.raw_injection(x_in)
        
        contexts = []
        for layer, down in self.downs:
            x = layer(x)
            contexts.append(x)
            x = down(x)
            
        x = self.mid_block(x)
        
        for up, reduce, layer in self.ups:
            x = up(x)
            x = torch.cat([x, contexts.pop()], dim=1) 
            x = reduce(x)
            x = layer(x)
            
        return self.final_conv(x)
