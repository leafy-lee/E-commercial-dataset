# --------------------------------------------------------
# Code inspired by Swin Transformer
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from .loss import KL_loss
from .saliency_detector import define_salD
from .basemodel import InvertedResidual
from torchvision import transforms
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from IPython import embed
from collections import OrderedDict
from itertools import product

'''
def initialize_gaussians(n_gaussians):
    """
    Return initialized Gaussian parameters.
    Dimensions: [idx, y/x, mu/logstd].
    """
    gaussians = torch.Tensor([
            list(product([0.25, 0.5, 0.75], repeat=2)) +
            [(0.5, 0.25), (0.5, 0.5), (0.5, 0.75)] +
            [(0.25, 0.5), (0.5, 0.5), (0.75, 0.5)] +
            [(0.5, 0.5)],
            [(-1.5, -1.5)] * 9 + [(0, -1.5)] * 3 + [(-1.5, 0)] * 3 +
            [(0, 0)],
        ]).permute(1, 2, 0)

    gaussians = nn.Parameter(gaussians, requires_grad=True)
    return gaussians


def make_gaussian_maps(x, gaussians, size=None, scaling=6.):
    """Construct prior maps from Gaussian parameters."""
    if size is None:
        size = x.shape[-2:]
        bs = x.shape[0]
    else:
        size = [size] * 2
        bs = 1
    dtype = x.dtype
    device = x.device

    gaussian_maps = []
    map_template = torch.ones(*size, dtype=dtype, device=device)
    meshgrids = torch.meshgrid(
        [torch.linspace(0, 1, size[0], dtype=dtype, device=device),
         torch.linspace(0, 1, size[1], dtype=dtype, device=device), ])

    for gaussian_idx, yx_mu_logstd in enumerate(torch.unbind(gaussians)):
        map = map_template.clone()
        for mu_logstd, mgrid in zip(yx_mu_logstd, meshgrids):
            mu = mu_logstd[0]
            std = torch.exp(mu_logstd[1])
            map *= torch.exp(-((mgrid - mu) / std) ** 2 / 2)

        map *= scaling
        gaussian_maps.append(map)

    gaussian_maps = torch.stack(gaussian_maps)
    gaussian_maps = gaussian_maps.unsqueeze(0).expand(bs, -1, -1, -1)
    return gaussian_maps


def get_gaussian_maps(x, **kwargs):
    """Return the constructed Gaussian prior maps."""
    gaussians = initialize_gaussians(16)
    gaussian_maps = make_gaussian_maps(x, gaussians, **kwargs)
    return gaussian_maps
'''

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


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

def get_saliency_attn(x, window_size):
    #print("looking for size", x.shape)
    B_, N, _ = x.shape
    attn = x @ x.transpose(-2, -1)
    #print(attn.size())
    softmax = nn.Softmax(dim=-1)
    attn = softmax(attn)
    return attn

def cal_attn_loss(attn, attn_gt):
    return KL_loss(attn,attn_gt)

def saliency_merging(pic):
    pooling = F.avg_pool2d(pic, kernel_size=2, stride=2)
    return pooling

def saliency_embedding(pic):
    pooling = F.avg_pool2d(pic, kernel_size=4, stride=4)
    B, C, H, W = pooling.shape
    # print("watching size", B, C, H, W)
    pooling = pooling.permute(0, 2, 3, 1)
    return pooling


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
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, saliency_gt=None, mask=None, need_loss=False):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        
        
        # for test
        attn_loss = 0



        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # print(qkv.shape)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
            
        # for test
        '''
        if need_loss:
            attn_gt = get_saliency_attn(saliency_gt, self.window_size).unsqueeze(1).expand(B_, self.num_heads, N, N)
            # print("attnshape","attn_gt shape", attn.shape, attn_gt.shape)
            attn_loss = cal_attn_loss(attn, attn_gt) / self.num_heads
        '''


        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if need_loss:
            return x, attn_loss  
        else:
            return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
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

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, require_attn_loss=False, shift_attn=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
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

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.require_attn_loss = require_attn_loss
        self.shift_attn = shift_attn

    def forward(self, x, saliency_gt):
        # print()
        # print("looking for saliency", saliency_gt.shape, x.shape)
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        
        
        # for test
        self.require_attn_loss = False



        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.require_attn_loss:
            saliency_gt = saliency_gt.view(B, H, W, 1)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        if self.require_attn_loss:
            if self.shift_attn:
                shifted_saliency_gt = torch.roll(saliency_gt, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_saliency_gt = saliency_gt

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        if self.require_attn_loss:
            saliency_gt_windows = window_partition(shifted_saliency_gt, self.window_size)
        # print("looking for window", saliency_gt_windows.shape, x_windows.shape)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        if self.require_attn_loss:
            saliency_gt_windows = saliency_gt_windows.view(-1, self.window_size * self.window_size, 1)

        # W-MSA/SW-MSAf
        if self.require_attn_loss:
            if not self.shift_attn:
                if self.shift_size > 0:
                    # print("self.shift_size > 0")
                    attn_windows = self.attn(x_windows, mask=self.attn_mask, need_loss=False)  # nW*B, window_size*window_size, C
                else:
                    # print("self.shift_size = 0 attn_loss")
                    attn_windows, attn_loss = self.attn(x_windows, saliency_gt=saliency_gt_windows, mask=self.attn_mask, need_loss=True)  # nW*B, window_size*window_size, C
                    # print(attn_loss)
            else:
                attn_windows, attn_loss = self.attn(x_windows, saliency_gt=saliency_gt_windows, mask=self.attn_mask, need_loss=True)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.attn_mask, need_loss=False)  # nW*B, window_size*window_size, C


        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        if self.require_attn_loss:
            saliency_gt_windows = saliency_gt_windows.view(-1, self.window_size, self.window_size, 1)
            shifted_saliency_gt = window_reverse(saliency_gt_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        if self.require_attn_loss:
            if self.shift_attn:
                saliency_gt = torch.roll(shifted_saliency_gt, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                saliency_gt = shifted_saliency_gt
                # print("calculating done")
                # print()

        x = x.view(B, H * W, C)
        if self.require_attn_loss:
            saliency_gt = saliency_gt.view(B, 1, H, W)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        if not self.shift_attn:
            if self.shift_size > 0:
                return x, saliency_gt, 0
        if self.require_attn_loss:
            return x, saliency_gt, attn_loss
        else:
            return x, saliency_gt, 0

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"



class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        """
        need to add merging opr
        """

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"




class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
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
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num = 1
       #  self.val = val

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 require_attn_loss=False,
                                 shift_attn=False)
            for i in range(depth-2)] + [
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if ((depth-2) % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[depth-2] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 require_attn_loss=False,
                                 shift_attn=False)
                                 ] + [
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if ((depth-1) % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[depth-1] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 require_attn_loss=True,
                                 shift_attn=True)
                                 ])
            

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, image, saliency_gt):
        # print(x[1].shape, "merging shape")
        cnt = 0
        ans_attn_loss = 0
        blkcnt = 0
        for blk in self.blocks:
            if self.use_checkpoint:
                image, saliency_gt, attn_loss = checkpoint.checkpoint(blk, image, saliency_gt)
            else:
                image, saliency_gt, attn_loss = blk(image, saliency_gt)
            if torch.is_tensor(attn_loss):
                ans_attn_loss += attn_loss
                # print("adding", attn_loss, ans_attn_loss, len(self.blocks), blkcnt)
                cnt += 1
            blkcnt += 1
            # print("detect loss from inside from", self.num, "layer, ", cnt, "block \n loss is", attn_loss)
        self.num += 1
        # print("finally", ans_attn_loss)

        if self.downsample is not None:
            image = self.downsample(image)
            # saliency_gt = saliency_merging(saliency_gt)
            # print(saliency_gt.shape, "merging shape")
        ans = ans_attn_loss/cnt if cnt else 0
        return image, saliency_gt, ans

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=768, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # print(x.shape)
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x



class SSwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
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
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=768, patch_size=4, in_chans=3, num_classes=1,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, head='denseNet_15layer', bn_momentum=0.01, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        #self.val = val

        # split image into non-overlapping patches
        # print("getting size", img_size, patch_size)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # print("building layer, i_layer", i_layer, "resolution", (patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer))),
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.salhead = define_salD(768, netSalD=head)
        '''
        self.upconv1 = double_conv(embed_dim * 8, embed_dim * 8, embed_dim * 4)
        self.upconv2 = double_conv(embed_dim * 4, embed_dim * 4, embed_dim * 2)
        self.upconv3 = double_conv(embed_dim * 2, embed_dim * 2, embed_dim)
        self.upconv4 = double_conv(embed_dim, embed_dim, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )
        # Initialize Post-CNN module with optional dropout
        post_cnn = [
            ('inv_res', InvertedResidual(
                embed_dim * 8 + 16,
                embed_dim * 8, 1, 1, bn_momentum=bn_momentum,
            ))
        ]
        self.post_cnn = nn.Sequential(OrderedDict(post_cnn))
        '''
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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

    def forward_features(self, image, saliency_gt):
        image = self.patch_embed(image)
        #print("image begins", image.shape)
        saliency_gt = saliency_embedding(saliency_gt)
        image = self.pos_drop(image)

        attn_loss_all = 0
        H, W = self.patches_resolution[0], self.patches_resolution[1]
        B, L, C = image.shape
        # layer_out_put = [image.transpose(1, 2).view(B, C, H, W)]
        i_layer = 0
        # print(image.shape)
        for layer in self.layers:
            # print("detect layer size")
            image, saliency_gt, attn_loss = layer(image, saliency_gt)
            # print(image.shape)
            i_layer = min(i_layer + 1, self.num_layers - 1)
            # print("detect loss from deeper inside", attn_loss, attn_loss_all)
            attn_loss_all += attn_loss
            H, W = self.patches_resolution[0] // (2 ** i_layer), self.patches_resolution[1] // (2 ** i_layer)
            B, L, C = image.shape
            # layer_out_put.append(image.transpose(1, 2).view(B, C, H, W))
        # print(attn_loss_all, "before")
        attn_loss_all = attn_loss_all / self.num_layers

        image = self.norm(image)
        
        # layer
        # layer_out_put[4] =  image.transpose(1, 2).view(B, C, H, W)  
        
        B, L, C = image.shape
        H, W = self.patches_resolution[0] // (2 ** (self.num_layers -1)), self.patches_resolution[1] // (2 ** (self.num_layers -1))
        image = image.view(B, H, W, C).permute(0,3,1,2)  # B C H W
        # gaussian_maps = get_gaussian_maps(image)
        # image = torch.cat((image, gaussian_maps), dim=1)
        # image = self.post_cnn(image)
        # print("looking for shape", B, H, W, C)
        # return image.permute(0,3,1,2), attn_loss_all, layer_out_put       
        # return image, attn_loss_all, layer_out_put
        return image, attn_loss_all


    def forward(self, image, saliency_gt):
        # feature, attn_loss, source = self.forward_features(image, saliency_gt)
        # print("imageshape", image.shape, saliency_gt.shape)
        feature, attn_loss = self.forward_features(image, saliency_gt)
        # print(attn_loss.shape)
        sal = self.salhead(feature)

        # refine feature for ocr
        sal_max = torch.amax(sal, (2,3))
        sal_ref = []
        for i in range(sal.size()[0]):
            sal_ref.append(sal[i:i+1,:,:,:]/(5*sal_max[i]) - 0.1)
        sal_refine = transforms.Resize(28)(torch.cat(sal_ref, dim=0))
        # print(sal_refine.shape, feature.shape)
        refine_ocr_feature = torch.mul((1 + sal_refine), source[4])
        refine_ocr_feature = source[4]
        
        y = torch.cat([refine_ocr_feature, source[3]], dim=1)
        y = self.upconv1(y)
        y = F.interpolate(y, size=source[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, source[2]], dim=1)
        y = self.upconv2(y)
        y = F.interpolate(y, size=source[1].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, source[1]], dim=1)
        y = self.upconv3(y)
        y = F.interpolate(y, size=source[0].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, source[0]], dim=1)
        ocr_fea = self.upconv4(y)
        y = self.conv_cls(ocr_fea)
        
        #print(torch.max(sal), torch.min(sal))
        #print(torch.max(y), torch.min(y))
        
        # refine feature for saliency
        y_sum = y[:,:1,:,:] + y[:,1:,:,:]
        y_max = torch.amax(y_sum, (2,3))
        y_ref = []
        for i in range(y_sum.size()[0]):
            y_ref.append(y_sum[i:i+1,:,:,:]/(5*y_max[i]) - 0.1)
        y_refine = transforms.Resize(28)(torch.cat(y_ref, dim=0))
        refine_sal_feature = torch.mul((1 + y_refine), feature)
        
        sal = self.salhead(refine_sal_feature)
        # return sal, attn_loss, y.permute(0, 2, 3, 1)
        # return sal, y.permute(0, 2, 3, 1)
        # print(sal.shape, attn_loss,"in model")
        # return sal, attn_loss
        return sal, attn_loss


