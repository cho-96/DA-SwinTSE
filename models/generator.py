# -----------------------------------------------------------------------------------
# SwinTSE
# Written by Haechan Cho
# -----------------------------------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def to_embed(x, embed_norm_layer=None):
    x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
    if embed_norm_layer is not None:
        x = embed_norm_layer(x)
    return x


def to_original(x, x_size):
    B, _, C = x.shape
    x = x.transpose(1, 2).view(B, C, x_size[0], x_size[1])  # B Ph*Pw C
    return x



class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)) 

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w])) 
        coords_flatten = torch.flatten(coords, 1) 
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] 
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
        relative_coords[:, :, 0] += self.window_size[0] - 1 
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1) 
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) 

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1) 
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() 
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1)) 
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size) 
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) 
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    

class DoubleSwinTransformerBlock(nn.Module):
    r""" Double Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., s_size=20, t_size=1, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = nn.ModuleList([
            WindowAttention(
                dim//2, window_size=to_2tuple(self.window_size), num_heads=num_heads//2,
                qk_scale=qk_scale, attn_drop=attn_drop),
            WindowAttention(
                dim//2, window_size=to_2tuple(self.window_size), num_heads=num_heads//2,
                qk_scale=qk_scale, attn_drop=attn_drop),
        ])

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.s_size = s_size
        self.t_size = t_size

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1)) 
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size) 
        ws = self.window_size
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) 
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        dy_m = self.s_size
        dt_s = self.t_size

        kmh2mps = 1000.0 / 3600.0
        cv_min = 60.0 * kmh2mps   
        cv_max = 100.0 * kmh2mps  
        cw_min = 10.0 * kmh2mps   
        cw_max = 25.0 * kmh2mps  

        device = attn_mask.device

        r = torch.arange(ws, device=device)
        c = torch.arange(ws, device=device)
        rr, cc = torch.meshgrid(r, c, indexing='ij')        
        coords = torch.stack([rr.reshape(-1), cc.reshape(-1)], dim=-1) 

        delta = coords[:, None, :] - coords[None, :, :]     
        dy = delta[..., 0].to(torch.float32) * dy_m
        dt = delta[..., 1].to(torch.float32) * dt_s

        eye = torch.eye(ws*ws, device=device, dtype=torch.bool)

        dt_pos = dt > 0
        dy_up  = dy >= 0
        speed  = torch.zeros_like(dy)
        speed[dt_pos] = dy[dt_pos] / dt[dt_pos]         
        allow_v = (dt_pos & dy_up & (speed >= cv_min) & (speed <= cv_max)) | eye

        cone_v = torch.zeros(ws*ws, ws*ws, device=device)
        cone_v[~allow_v] = -100.0                     

        dy_down = dy <= 0
        speed_w = torch.zeros_like(dy)
        speed_w[dt_pos] = torch.abs(dy[dt_pos]) / dt[dt_pos]
        allow_w = (dt_pos & dy_down & (speed_w >= cw_min) & (speed_w <= cw_max)) | eye

        cone_w = torch.zeros(ws*ws, ws*ws, device=device)
        cone_w[~allow_w] = -100.0

        cone_v = cone_v.unsqueeze(0)   
        cone_w = cone_w.unsqueeze(0)

        attn_mask1 = attn_mask + cone_v  
        attn_mask2 = attn_mask + cone_w  

        return attn_mask1, attn_mask2

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        x1 = x[:, :, :, : C // 2]
        x2 = x[:, :, :, C // 2 :]

        if self.shift_size > 0:
            shifted_x1 = torch.roll(x1, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_x2 = torch.roll(x2, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x1 = x1
            shifted_x2 = x2

        x_windows1 = window_partition(shifted_x1, self.window_size)
        x_windows1 = x_windows1.view(-1, self.window_size * self.window_size, C // 2)

        x_windows2 = window_partition(shifted_x2, self.window_size)
        x_windows2 = x_windows2.view(-1, self.window_size * self.window_size, C // 2)

        attn_mask1, attn_mask2 = self.calculate_mask(x_size)

        attn_windows1 = self.attn[0](x_windows1, mask=attn_mask1.to(x.device))
        attn_windows2 = self.attn[1](x_windows2, mask=attn_mask2.to(x.device))

        attn_windows1 = attn_windows1.view(-1, self.window_size, self.window_size, C//2)
        shifted_x1 = window_reverse(attn_windows1, self.window_size, H, W)

        attn_windows2 = attn_windows2.view(-1, self.window_size, self.window_size, C//2)
        shifted_x2 = window_reverse(attn_windows2, self.window_size, H, W)

        if self.shift_size > 0:
            x1 = torch.roll(shifted_x1, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x2 = torch.roll(shifted_x2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x1 = shifted_x1
            x2 = shifted_x2
        x = torch.cat([x1.reshape(B, H * W, C // 2), x2.reshape(B, H * W, C // 2)], dim=2)

        x = shortcut + self.mlp(self.norm2(x))

        return x


class DoubleBasicLayer(nn.Module):
    """ A basic Double Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., s_size=20, t_size=1, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):

        super().__init__()
        self.dim = dim
        self.depth = depth


        self.blocks = nn.ModuleList([
            DoubleSwinTransformerBlock(dim=dim,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 s_size=20,
                                 t_size=1,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        
    def forward(self, x, x_size):
        for blk in self.blocks:
            x = blk(x, x_size)
        return x

class GCL(nn.Module):
    def __init__(self, dim, bias):
        super(GCL, self).__init__()

        hidden_features = int(dim)

        self.feat_conv = nn.Conv2d(dim, dim, 3, padding=1, bias=bias)
        self.mask_conv = nn.Conv2d(1, dim, 3, padding=1, bias=False)

        self.gate_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x, mask):
        feat = self.feat_conv(x)
        mask_feat = self.mask_conv(mask)

        gate = torch.sigmoid(self.gate_conv(feat + mask_feat))
        x = gate * feat

        return self.project_out(x)


class Double_STB(nn.Module):
    """Swin Transformer Block (STB).

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., s_size=20, t_size=1, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, embed_norm=True):
        super(Double_STB, self).__init__()

        self.dim = dim
        self.embed_norm_layer=norm_layer(dim) if embed_norm else None

        self.residual_group = DoubleBasicLayer(dim=dim,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         s_size=20,
                                         t_size=1,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer)
            
        self.gate = GCL(dim=dim, bias=False)

    def forward(self, x, mask, x_size):
        return to_embed(self.gate(to_original(self.residual_group(x, x_size), x_size), mask), self.embed_norm_layer) + x


class DASwinTSE(nn.Module):
    r""" 

    Args:
        input_size (int | tuple(int)): Input size. Default 64
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        pe (str): If True, add position embedding to the patch embedding. Default: False
        embed_norm (bool): If True, add normalization after embedding. Default: True
    """

    def __init__(self, in_chans=3,out_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., s_size = 20, t_size = 1, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, pe=None, embed_norm=True, 
                 **kwargs):
        super(DASwinTSE, self).__init__()
        num_in_ch = in_chans
        self.outc_chans=out_chans

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.num_layers = len(depths)
        self.pe = pe
        self.mlp_ratio = mlp_ratio
        self.embed_norm_layer=norm_layer(embed_dim) if embed_norm else None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = Double_STB(dim=embed_dim,
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         s_size = s_size,
                         t_size = t_size,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         embed_norm = embed_norm
                         )
            self.layers.append(layer)
        self.norm = norm_layer(embed_dim)

        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim+num_in_ch, out_chans, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.1)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, mask):
        x_size = (x.shape[2], x.shape[3])
        x = to_embed(x, self.embed_norm_layer)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, mask, x_size)

        x = self.norm(x) 
        x = to_original(x, x_size)

        return x

    def forward(self, x, mask):
        x_first = self.conv_first(x)
        res = self.conv_after_body(self.forward_features(x_first, mask)) + x_first
        x = self.conv_last(torch.cat((res,x),1))
        return x