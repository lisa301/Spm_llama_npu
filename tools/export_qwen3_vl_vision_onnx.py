#!/usr/bin/env python3
# Export Qwen3-VL vision tower (ViT + merger) to ONNX with a static input shape.
#
# This matches the Rust implementation in:
# - spm-core/src/models/qwen3_vl/vision.rs
#
# Notes:
# - Input expects normalized RGB float32 tensor: (1, 3, H, W) where H=W=--side.
#   Normalization should be done outside the model: (x/255 - 0.5) / 0.5
# - Qwen3-VL vision uses a temporal_patch_size (often 2). The Rust code duplicates
#   the same image along channel dim. We replicate that here inside the graph.

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from safetensors.torch import safe_open


@dataclass(frozen=True)
class VisionCfg:
    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    num_position_embeddings: int
    spatial_merge_size: int
    out_hidden_size: int


def load_vision_cfg(model_dir: str) -> VisionCfg:
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    vc = cfg["vision_config"]
    return VisionCfg(
        depth=int(vc["depth"]),
        hidden_size=int(vc["hidden_size"]),
        intermediate_size=int(vc["intermediate_size"]),
        num_heads=int(vc["num_heads"]),
        in_channels=int(vc["in_channels"]),
        patch_size=int(vc["patch_size"]),
        temporal_patch_size=int(vc["temporal_patch_size"]),
        num_position_embeddings=int(vc["num_position_embeddings"]),
        spatial_merge_size=int(vc["spatial_merge_size"]),
        out_hidden_size=int(vc["out_hidden_size"]),
    )


class VitAttention(nn.Module):
    def __init__(self, hidden: int, num_heads: int):
        super().__init__()
        assert hidden % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=True)
        self.proj = nn.Linear(hidden, hidden, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, seq, hidden)
        b, seq, hidden = x.shape
        qkv = self.qkv(x)  # (b, seq, 3*hidden)
        qkv = qkv.view(b, seq, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2).contiguous()  # (b, h, seq, d)
        k = qkv[:, :, 1].transpose(1, 2).contiguous()
        v = qkv[:, :, 2].transpose(1, 2).contiguous()

        # Match Rust: do attention math in fp32 and cast back.
        in_dtype = q.dtype
        q = q.float()
        k = k.float()
        v = v.float()
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = torch.softmax(att, dim=-1)
        y = torch.matmul(att, v).to(in_dtype)

        y = y.transpose(1, 2).contiguous().view(b, seq, hidden)
        return self.proj(y)


class VitMlp(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, intermediate, bias=True)
        self.fc2 = nn.Linear(intermediate, hidden, bias=True)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VitBlock(nn.Module):
    def __init__(self, cfg: VisionCfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=1e-5, elementwise_affine=True)
        self.attn = VitAttention(cfg.hidden_size, cfg.num_heads)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=1e-5, elementwise_affine=True)
        self.mlp = VitMlp(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Merger(nn.Module):
    def __init__(self, token_hidden: int, spatial_merge_size: int, out_hidden: int):
        super().__init__()
        m = int(spatial_merge_size)
        assert m >= 1
        self.spatial_merge_size = m
        merged_hidden = token_hidden * m * m
        self.norm = nn.LayerNorm(token_hidden, eps=1e-5, elementwise_affine=True)
        self.fc1 = nn.Linear(merged_hidden, merged_hidden, bias=True)
        self.fc2 = nn.Linear(merged_hidden, out_hidden, bias=True)
        self.act = nn.GELU(approximate="tanh")

    def spatial_merge(self, x: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
        # x: (b, seq, hidden)
        b, _seq, hidden = x.shape
        m = self.spatial_merge_size
        if m == 1:
            return x
        assert hp % m == 0 and wp % m == 0

        # Match Rust reshape/transpose order exactly.
        x = x.view(b, hp, wp, hidden)
        x = x.view(b, hp // m, m, wp // m, m, hidden)
        x = x.transpose(2, 3).contiguous()
        return x.view(b, (hp // m) * (wp // m), hidden * m * m)

    def forward(self, x: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
        x = self.norm(x)
        x = self.spatial_merge(x, hp, wp)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class VisionTower(nn.Module):
    def __init__(self, cfg: VisionCfg, side: int):
        super().__init__()
        self.cfg = cfg
        self.side = int(side)
        if self.side % cfg.patch_size != 0:
            raise ValueError(f"side {side} not divisible by patch_size {cfg.patch_size}")
        self.hp = self.side // cfg.patch_size
        self.wp = self.side // cfg.patch_size
        m = max(1, cfg.spatial_merge_size)
        if self.hp % m != 0 or self.wp % m != 0:
            raise ValueError(
                f"patch grid {self.hp}x{self.wp} not divisible by spatial_merge_size {m}"
            )
        base_side = int(math.isqrt(cfg.num_position_embeddings))
        if base_side * base_side != cfg.num_position_embeddings:
            raise ValueError("num_position_embeddings is not a square")
        if self.hp > base_side or self.wp > base_side:
            raise ValueError(f"patch grid {self.hp}x{self.wp} exceeds base {base_side}x{base_side}")

        # Precompute pos ids (top-left subset of a base_side x base_side grid).
        pos_ids = []
        for r in range(self.hp):
            for c in range(self.wp):
                pos_ids.append(r * base_side + c)
        self.register_buffer("pos_ids", torch.tensor(pos_ids, dtype=torch.long), persistent=False)

        # Patch embedding conv: weights loaded from [out, in, temporal, kh, kw] reshaped to [out, in*temporal, kh, kw].
        self.patch = nn.Conv2d(
            cfg.in_channels * cfg.temporal_patch_size,
            cfg.hidden_size,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            bias=True,
        )
        self.pos_embed = nn.Embedding(cfg.num_position_embeddings, cfg.hidden_size)
        self.blocks = nn.ModuleList([VitBlock(cfg) for _ in range(cfg.depth)])
        self.merger = Merger(cfg.hidden_size, cfg.spatial_merge_size, cfg.out_hidden_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # image: (1, 3, side, side) normalized float32
        x = image
        t = int(self.cfg.temporal_patch_size)
        if t > 1:
            # Match Rust: duplicate the same frame t times along channel dim.
            x = torch.cat([x] * t, dim=1)
        x = self.patch(x)  # (1, hidden, hp, wp)
        x = x.flatten(2).transpose(1, 2).contiguous()  # (1, seq, hidden)

        pos = self.pos_embed(self.pos_ids).unsqueeze(0)  # (1, seq, hidden)
        x = x + pos

        for blk in self.blocks:
            x = blk(x)

        x = self.merger(x, self.hp, self.wp)  # (1, (hp/m)*(wp/m), out_hidden)
        return x


def load_weights(model: VisionTower, st_path: str) -> None:
    # Map safetensors keys to torch module params.
    sd: Dict[str, torch.Tensor] = {}
    with safe_open(st_path, framework="pt", device="cpu") as f:
        # Patch embed
        w5 = f.get_tensor("model.visual.patch_embed.proj.weight")  # (out, in, t, kh, kw)
        b = f.get_tensor("model.visual.patch_embed.proj.bias")
        out, inc, t, kh, kw = w5.shape
        sd["patch.weight"] = w5.reshape(out, inc * t, kh, kw).contiguous()
        sd["patch.bias"] = b

        # Pos embed
        sd["pos_embed.weight"] = f.get_tensor("model.visual.pos_embed.weight")

        # Blocks
        for i in range(model.cfg.depth):
            p = f"model.visual.blocks.{i}."
            sd[f"blocks.{i}.norm1.weight"] = f.get_tensor(p + "norm1.weight")
            sd[f"blocks.{i}.norm1.bias"] = f.get_tensor(p + "norm1.bias")
            sd[f"blocks.{i}.attn.qkv.weight"] = f.get_tensor(p + "attn.qkv.weight")
            sd[f"blocks.{i}.attn.qkv.bias"] = f.get_tensor(p + "attn.qkv.bias")
            sd[f"blocks.{i}.attn.proj.weight"] = f.get_tensor(p + "attn.proj.weight")
            sd[f"blocks.{i}.attn.proj.bias"] = f.get_tensor(p + "attn.proj.bias")
            sd[f"blocks.{i}.norm2.weight"] = f.get_tensor(p + "norm2.weight")
            sd[f"blocks.{i}.norm2.bias"] = f.get_tensor(p + "norm2.bias")
            sd[f"blocks.{i}.mlp.fc1.weight"] = f.get_tensor(p + "mlp.linear_fc1.weight")
            sd[f"blocks.{i}.mlp.fc1.bias"] = f.get_tensor(p + "mlp.linear_fc1.bias")
            sd[f"blocks.{i}.mlp.fc2.weight"] = f.get_tensor(p + "mlp.linear_fc2.weight")
            sd[f"blocks.{i}.mlp.fc2.bias"] = f.get_tensor(p + "mlp.linear_fc2.bias")

        # Merger
        sd["merger.norm.weight"] = f.get_tensor("model.visual.merger.norm.weight")
        sd["merger.norm.bias"] = f.get_tensor("model.visual.merger.norm.bias")
        sd["merger.fc1.weight"] = f.get_tensor("model.visual.merger.linear_fc1.weight")
        sd["merger.fc1.bias"] = f.get_tensor("model.visual.merger.linear_fc1.bias")
        sd["merger.fc2.weight"] = f.get_tensor("model.visual.merger.linear_fc2.weight")
        sd["merger.fc2.bias"] = f.get_tensor("model.visual.merger.linear_fc2.bias")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="HF-like model directory containing config.json and model.safetensors")
    ap.add_argument("--safetensors", default=None, help="Path to model.safetensors (defaults to <model-dir>/model.safetensors)")
    ap.add_argument("--side", type=int, default=448, help="Fixed square input side length in pixels (e.g. 448 for seq=784 when patch_size=16)")
    ap.add_argument("--out", required=True, help="Output ONNX path")
    # TPU-MLIR toolchains are often more compatible with lower opsets.
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    cfg = load_vision_cfg(args.model_dir)
    st_path = args.safetensors or os.path.join(args.model_dir, "model.safetensors")
    if not os.path.exists(st_path):
        raise FileNotFoundError(st_path)

    model = VisionTower(cfg, side=args.side).eval()
    load_weights(model, st_path)

    # Export in fp32 first; you can later quantize/convert for BM1684.
    model = model.float()
    dummy = torch.randn(1, 3, args.side, args.side, dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["image"],
        output_names=["vision_embeds"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print("exported:", args.out)
    print(f"input: (1,3,{args.side},{args.side})")
    print(f"patch_grid: {model.hp}x{model.wp} seq={model.hp*model.wp}")
    m = max(1, cfg.spatial_merge_size)
    print(f"output: (1,{(model.hp//m)*(model.wp//m)},{cfg.out_hidden_size})")


if __name__ == "__main__":
    main()
