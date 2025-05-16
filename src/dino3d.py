# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

import torch.nn.functional as F
import torch
import torch.nn as nn
from timm.layers import DropPath, to_2tuple, to_3tuple


def load_checkpoint(
    model,
    pretrained_weights=None,
    checkpoint_key="teacher",
    map_location="cpu",
    bootstrap_method="centering",
):


    state_dict = torch.load(pretrained_weights, map_location=map_location)
    

    if checkpoint_key is not None and checkpoint_key in state_dict:
        #print(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]

    # --- starting inflate/center weights ---
    n_slices = model.patch_embed.patch_size[-1]
    n_chans = model.patch_embed.in_chans
    #print("number of slices and number of channels", n_slices,n_chans)
    key = "patch_embed.proj.weight"
    emb = state_dict[key]
    #print("key and emb(i assume embessings )", key)
    #print("Old:", emb.shape, emb.sum())

    emb = emb.sum(1, keepdim=True)  # from colored to grayed
    #print(" emb after summing the channels",emb.shape)
    
    emb = emb.repeat(1, model.patch_embed.in_chans, 1, 1) / model.patch_embed.in_chans
    #print("After grayscale + channel repeat:", emb.shape)  # [768, 1, 16, 16]
    # from 1-channel grayed to n-channel grayed
    # Interpolate from (16,16) → (1,50) to match H×W in patch size
    emb = F.interpolate(emb, size=(3, 40), mode='bicubic', align_corners=False)
    #print("After depth interpolation:", emb.shape)
    # Convert to 3D by inflating in the depth dimension
    depth = model.patch_embed.patch_size[0]  # 16
    emb = emb.unsqueeze(2).repeat(1, 1, depth, 1, 1)  # [768, 1, 16, 16, 16]
    #print("After depth inflation:", emb.shape)
    
    if bootstrap_method == "inflation":
        print("Using inflation strategy")
        emb = emb / depth  # normalize across slices
    elif bootstrap_method == "centering":
        print("Using centering strategy")
        center_idx = depth // 2
        for i in range(depth):
            if i != center_idx:
                emb[:, :, i, :, :] = 0
    else:
        raise ValueError("Invalid bootstrap method")

    #print("New:", emb.shape, emb.sum())
    state_dict[key] = emb
    # --- ending inflate/center weights ---

    ori_num_patches = state_dict["pos_embed"].shape[1] - 1
    cur_num_patches = model.patch_embed.num_patches
    print("ori_num_patches and cur_num_patches", ori_num_patches, cur_num_patches)
    

    if ori_num_patches != cur_num_patches:
        print("Resizing position embedding from", ori_num_patches, "to", cur_num_patches)
        emb = state_dict["pos_embed"]
        cls_emb = emb[:, 0:1]        # (1, 1, dim)
        patch_emb = emb[:, 1:]       # (1, 196, dim)

        # Interpolate in 2D
        dim = patch_emb.shape[-1]
        side = int(math.sqrt(ori_num_patches))
        patch_emb = patch_emb.reshape(1, side, side, dim).permute(0, 3, 1, 2)
        h_new = model.img_size[1] // model.patch_embed.patch_size[1]
        w_new = model.img_size[2] // model.patch_embed.patch_size[2]
        print("patch_embed h_new and w_new", patch_emb.shape, h_new, w_new)
        patch_emb = F.interpolate(patch_emb, size=(h_new, w_new), mode='bicubic', align_corners=False)
        patch_emb = patch_emb.permute(0, 2, 3, 1).reshape(1, -1, dim)
        print("patch_embed after interpolation", patch_emb.shape)
        # Replicate to depth
        d_new = model.img_size[0] // model.patch_embed.patch_size[0]
        patch_emb = patch_emb.unsqueeze(1).repeat(1, d_new, 1, 1).reshape(1, -1, dim)

        # Combine with CLS token
        state_dict["pos_embed"] = torch.cat([cls_emb, patch_emb], dim=1)

    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print(
        "Pretrained weights found at {} and loaded with msg: {}".format(
            pretrained_weights, msg
        )
    )



def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.25,
    ):
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


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.1,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.25,
        attn_drop=0.1,
        drop_path=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()

        if isinstance(patch_size, int):
            patch_size = to_3tuple(patch_size)
            
        if isinstance(img_size, int):
            img_size = to_3tuple(img_size)
        """
        if type(img_size) == int:
            num_patches = (img_size // patch_size[0]) * (img_size // patch_size[0])
        else:
            num_patches = (img_size[0] // patch_size[0]) * (
                img_size[0] // patch_size[0]
            )
        """
        #print("img_size and patch_size", img_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        #print("img_size[0] and img_size[1] img_size[2]", img_size[0], img_size[1],img_size[2])
        #print("patch_size[0] and patch_size[1] patch_size[2]", patch_size[0], patch_size[1],patch_size[2])
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.in_chans = in_chans
        #print("num_patches", self.num_patches)    
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        #x = x.permute(0, 1, 3, 4, 2)
        #print("after permute", x.shape)
        x = self.proj(x)
        #print("after conv", x.shape)
        #x = x.squeeze(2)
        x = x.flatten(2).transpose(1, 2)
        #print("afer conv embed",x.shape)
        return x


class VisionTransformer3D(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=[224],
        patch_size=(16, 1, 50),
        in_chans=3,
        num_classes=2,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.25,
        attn_drop_rate=0.2,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        #print("in VT num_features", self.num_features)
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size= patch_size,
            in_chans= in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        #print("number of patches after patchembedd in VT" , num_patches)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        """
        # Classifier head
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        """
        # Classifier head
        
        self.head = nn.Sequential(
            nn.Linear(embed_dim , 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 3)  # Final classification layer
        )
     
        

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        # self.apply(self._init_weights)

    def init_weights(
        self,
        pretrained= "/home/shubham/D1/ViT/eeg_classification/dino/pretrained/dino_vitb16.pth",
        bootstrap_method="centering",
    ):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            load_checkpoint(
                self,
                pretrained_weights=pretrained,
                bootstrap_method=bootstrap_method,
                checkpoint_key="teacher",
            )
            
        elif pretrained is None:
            print("No pretrained weights provided, initializing from scratch.")
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")
        
    """
    def interpolate_pos_encoding(self, x_patch):
        
        #Interpolates 2D pretrained positional embeddings (H x W) and repeats across depth (D).
        #Assumes: pos_embed is [1, N+1, C], with CLS token at [:,0,:].
      
        num_patches_current = x_patch.shape[1]
        num_patches_pretrained = self.pos_embed.shape[1] - 1  # exclude CLS
        class_pos_embed = self.pos_embed[:, 0:1, :]  # [1, 1, C]
        patch_pos_embed = self.pos_embed[:, 1:, :]   # [1, N, C]
        dim = patch_pos_embed.shape[-1]

        # Get pretrained H and W (2D patch grid)
        H_pre = self.img_size[1] // self.patch_embed.patch_size[1]
        W_pre = self.img_size[2] // self.patch_embed.patch_size[2]
        assert H_pre * W_pre == num_patches_pretrained, "Mismatch in pretrained pos_embed shape!"

        # Reshape to [1, C, H, W]
        patch_pos_embed = patch_pos_embed.reshape(1, H_pre, W_pre, dim).permute(0, 3, 1, 2)  # [1, C, H, W]

        # Get target H and W from current input
        dummy_input = torch.zeros(1, self.patch_embed.in_chans, *self.img_size).to(x_patch.device)
        with torch.no_grad():
            patch_out = self.patch_embed.proj(dummy_input)
            _, _, D_new, H_new, W_new = patch_out.shape

        # Interpolate over H and W
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(H_new, W_new),
            mode='bicubic',
            align_corners=False
        )  # [1, C, H_new, W_new]

        # Repeat over D
        patch_pos_embed = patch_pos_embed.unsqueeze(2).repeat(1, 1, D_new, 1, 1)  # [1, C, D_new, H_new, W_new]

        # Reshape to [1, D*H*W, C]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).reshape(1, -1, dim)

        # Concat CLS token
        full_pos_embed = torch.cat((class_pos_embed, patch_pos_embed), dim=1)

        # Sanity check
        assert full_pos_embed.shape[1] == x_patch.shape[1] + 1, \
            f"Token count mismatch: got {full_pos_embed.shape[1]}, expected {x_patch.shape[1] + 1}"

        return full_pos_embed






   
   
    def interpolate_pos_encoding(self, x, D, H, W):

        #Interpolates 2D pretrained pos_embed → 3D for current (D, H, W) using patch size.
        #Args:
        #    x: Input tokens of shape [B, N+1, C]
        #    D, H, W: spatial input dimensions before patching (not after)
        #Returns:
        #    pos_embed interpolated to match token count
    
        num_patches_current = x.shape[1] - 1  # exclude CLS token
        num_patches_pretrained = self.pos_embed.shape[1] - 1
        print("num_patches_current and num_patches_pretrained", num_patches_current, num_patches_pretrained)
        if num_patches_current == num_patches_pretrained:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0:1, :]  # [1, 1, C]
        patch_pos_embed = self.pos_embed[:, 1:, :]   # [1, N, C]
        dim = patch_pos_embed.shape[-1]
        print("class_pos_embed, patch_pos_embed, dim", class_pos_embed.shape, patch_pos_embed.shape, dim)
        # Infer original patch grid size from model (the one used during pretraining)
        print("img_size[1] and img_size[2]", self.img_size[1], self.img_size[2])
        print("patch_embed.patch_size[1] and patch_embed.patch_size[2]", self.patch_embed.patch_size[1], self.patch_embed.patch_size[2])
        H_pretrained = self.img_size[1] // self.patch_embed.patch_size[1]
        W_pretrained = self.img_size[2] // self.patch_embed.patch_size[2]
        print("H_pretrained and W_pretrained", H_pretrained, W_pretrained)
        # Reshape and interpolate
        patch_pos_embed = patch_pos_embed.reshape(1, H_pretrained, W_pretrained, dim).permute(0, 3, 1, 2)
        print("patch_pos_embed after reshape", patch_pos_embed.shape)
        H_new = H // self.patch_embed.patch_size[1]
        W_new = W // self.patch_embed.patch_size[2]
        patch_pos_embed = F.interpolate(patch_pos_embed, size=(H_new, W_new), mode='bicubic', align_corners=False)
        print("patch_pos_embed after interpolation", patch_pos_embed.shape)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        print("patch_pos_embed after permute", patch_pos_embed.shape)
        D_new = D // self.patch_embed.patch_size[0]
        patch_pos_embed = patch_pos_embed.unsqueeze(1).repeat(1, D_new, 1, 1).reshape(1, -1, dim)
        full_pos_embed = torch.cat((class_pos_embed, patch_pos_embed), dim=1)
        print("full_pos_embed after cat", full_pos_embed.shape)
        return full_pos_embed
    """
    """
    #to work with any ractangualre shape 
    def interpolate_pos_encoding(self, x, D, H, W):
   
        #Interpolates 2D pretrained pos_embed → 3D positional encoding for current input (D, H, W).
        #Assumes self.pos_embed is [1, N+1, dim] and first token is CLS.
   
        num_patches_current = x.shape[1] - 1  # exclude CLS token
        num_patches_pretrained = self.pos_embed.shape[1] - 1
        print("num_patches_current and num_patches_pretrained", num_patches_current, num_patches_pretrained)
        if num_patches_current == num_patches_pretrained:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0:1, :]  # [1, 1, dim]
        patch_pos_embed = self.pos_embed[:, 1:, :]   # [1, N, dim]
        dim = patch_pos_embed.shape[-1]
        print("class_pos_embed, patch_pos_embed, dim", class_pos_embed.shape, patch_pos_embed.shape, dim)
        # --- NEW: Infer H_pretrained and W_pretrained from actual patch count and patch sizes ---
        patch_h = self.patch_embed.patch_size[1]
        patch_w = self.patch_embed.patch_size[2]
        print("patch_h and patch_w", patch_h, patch_w)
        # Find the best (H, W) match such that H * W = num_patches_pretrained
        # Assume you trained with H_pretrained and W_pretrained matching original input
        H_pretrained = int(round((num_patches_pretrained * patch_h / patch_w) ** 0.5))
        W_pretrained = num_patches_pretrained // H_pretrained
        print("H_pretrained and W_pretrained", H_pretrained, W_pretrained)
        assert H_pretrained * W_pretrained == num_patches_pretrained, \
            f"Cannot reshape pos_embed: {H_pretrained}x{W_pretrained} patches != {num_patches_pretrained}"

        print(">> Pretrained patch grid:", H_pretrained, W_pretrained)

        # Reshape to [1, dim, H, W]
        patch_pos_embed = patch_pos_embed.reshape(1, H_pretrained, W_pretrained, dim).permute(0, 3, 1, 2)
        print("patch_pos_embed after reshape", patch_pos_embed.shape)
        # Interpolate H and W
        H_new = H // patch_h
        W_new = W // patch_w
        print(">> Target patch grid:", H_new, W_new)

        patch_pos_embed = F.interpolate(
            patch_pos_embed, size=(H_new, W_new), mode='bicubic', align_corners=False
        )
        print("patch_pos_embed after interpolation", patch_pos_embed.shape)
        # Flatten and reshape to [1, new_patches, dim]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)

        # Expand across depth
        D_new = D // self.patch_embed.patch_size[0]
        print("D_new", D_new)
        patch_pos_embed = patch_pos_embed.unsqueeze(1).repeat(1, D_new, 1, 1).reshape(1, -1, dim)
        print("patch_pos_embed after unsqueeze and reshape", patch_pos_embed.shape)
        # Concatenate with CLS token
        full_pos_embed = torch.cat((class_pos_embed, patch_pos_embed), dim=1)
        print("full_pos_embed after cat", full_pos_embed.shape)
        return full_pos_embed
    """

    
    def prepare_tokens(self, x):
        
        # B, nc, w, h = x.shape
        #print("x shape before patch embedding", x.shape)
        B, c, d, h, w = x.shape
        #print("x shape after assignment ", x.shape)
        x = self.patch_embed(x)  # patch linear embedding
        #print("x after patch embedding", x.shape)
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        #print("x after cat", x.shape)
        # add positional encoding to each token
        #x = x + self.interpolate_pos_encoding(x, D=d, H=h, W=w)
        x = x + self.pos_embed[:, :x.shape[1]]

        #print("x after pos embed", x.shape)

        return self.pos_drop(x)
        """
        B, C, D, H, W = x.shape
        x_patch = self.patch_embed(x)  # Shape: [B, N, C]
        print("x_patch shape after patch embedding", x_patch.shape)
        # Interpolate positional encodings to match x_patch
        pos_embed = self.interpolate_pos_encoding(x_patch)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x_patch), dim=1)
        x = x + pos_embed
        return self.pos_drop(x)
    """
    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # <-- CLS embedding of shape [B, 768]


        #return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output
"""
if __name__ == "__main__":
    force_2d = False
    img_size = (160, 19, 500)
    patch_size = (20, 10, 50) #sh commented
    bootstrap_method = "centering"
    encoder = VisionTransformer3D(
        img_size=img_size, #if not force_2d else (1, img_size[1], img_size[2]), #sh commented 
        patch_size=patch_size,
        #if not force_2d  #sh commented 
        #else (patch_size_1, patch_size_2, 1), # sh commented
        embed_dim=768,
        depth=12,
        in_chans=1,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    encoder.init_weights(bootstrap_method=bootstrap_method)
    x = torch.randn(4, 1, 160, 19, 500)
    print("shape of x (input)", x)
    out = encoder(x)
    print([y.shape for y in out])
    print("output shape", out.shape)
    """