//! Qwen3-VL vision encoder (ViT-like) and projector (merger).
//!
//! This is a pragmatic implementation aimed at the Qwen3-VL safetensors layout:
//! - `model.visual.patch_embed.proj.(weight|bias)`
//! - `model.visual.pos_embed.weight`
//! - `model.visual.blocks.{i}.(norm1|attn|norm2|mlp).*`
//! - `model.visual.merger.*` (projects to language hidden size)

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Activation, Conv2d, Conv2dConfig, Embedding, LayerNorm, Linear, Module, VarBuilder};

use super::VisionConfig;

#[derive(Debug, Clone)]
struct VitAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VitAttention {
    fn load(vb: VarBuilder, hidden: usize, num_heads: usize) -> candle_core::Result<Self> {
        let head_dim = hidden / num_heads;
        Ok(Self {
            qkv: candle_nn::linear(hidden, 3 * hidden, vb.pp("qkv"))?,
            proj: candle_nn::linear(hidden, hidden, vb.pp("proj"))?,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, seq, hidden) = x.dims3()?;
        let qkv = self.qkv.forward(x)?; // (b, seq, 3*hidden)
        let qkv = qkv.reshape((b, seq, 3, self.num_heads, self.head_dim))?;
        let q = qkv.narrow(2, 0, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?; // (b, h, seq, d)
        let k = qkv.narrow(2, 1, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;

        let y = {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = candle_nn::ops::softmax(&att, D::Minus1)?;
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?
        };

        let y = y.transpose(1, 2)?.reshape((b, seq, hidden))?;
        self.proj.forward(&y)
    }
}

#[derive(Debug, Clone)]
struct VitMlp {
    fc1: Linear,
    fc2: Linear,
    act: Activation,
}

impl VitMlp {
    fn load_with_dims(vb: VarBuilder, hidden: usize, intermediate: usize) -> candle_core::Result<Self> {
        Ok(Self {
            fc1: candle_nn::linear(hidden, intermediate, vb.pp("linear_fc1"))?,
            fc2: candle_nn::linear(intermediate, hidden, vb.pp("linear_fc2"))?,
            act: Activation::GeluPytorchTanh,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.fc2.forward(&self.act.forward(&self.fc1.forward(x)?)?)
    }
}

#[derive(Debug, Clone)]
struct VitBlock {
    norm1: LayerNorm,
    attn: VitAttention,
    norm2: LayerNorm,
    mlp: VitMlp,
}

impl VitBlock {
    fn load(vb: VarBuilder, cfg: &VisionConfig) -> candle_core::Result<Self> {
        Ok(Self {
            norm1: candle_nn::layer_norm(cfg.hidden_size, 1e-5, vb.pp("norm1"))?,
            attn: VitAttention::load(vb.pp("attn"), cfg.hidden_size, cfg.num_heads)?,
            norm2: candle_nn::layer_norm(cfg.hidden_size, 1e-5, vb.pp("norm2"))?,
            mlp: VitMlp::load_with_dims(vb.pp("mlp"), cfg.hidden_size, cfg.intermediate_size)?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = (self.attn.forward(&self.norm1.forward(x)?)? + x)?;
        let x = (self.mlp.forward(&self.norm2.forward(&x)?)? + &x)?;
        Ok(x)
    }
}

#[derive(Debug, Clone)]
struct Merger {
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    act: Activation,
    spatial_merge_size: usize,
}

impl Merger {
    fn load(
        vb: VarBuilder,
        token_hidden: usize,
        spatial_merge_size: usize,
        out_hidden: usize,
    ) -> candle_core::Result<Self> {
        let merged_hidden = token_hidden * spatial_merge_size * spatial_merge_size;
        Ok(Self {
            // Note: the Qwen3-VL safetensors layout stores merger.norm over the per-patch hidden
            // (before spatial merging), while the fc layers operate on merged hidden.
            norm: candle_nn::layer_norm(token_hidden, 1e-5, vb.pp("norm"))?,
            fc1: candle_nn::linear(merged_hidden, merged_hidden, vb.pp("linear_fc1"))?,
            fc2: candle_nn::linear(merged_hidden, out_hidden, vb.pp("linear_fc2"))?,
            act: Activation::GeluPytorchTanh,
            spatial_merge_size,
        })
    }

    fn spatial_merge(&self, x: &Tensor, hp: usize, wp: usize) -> candle_core::Result<Tensor> {
        let (b, seq, hidden) = x.dims3()?;
        if seq != hp * wp {
            return Err(candle_core::Error::msg(format!(
                "spatial_merge: seq {} != hp*wp {}*{}",
                seq, hp, wp
            )));
        }
        let m = self.spatial_merge_size;
        if m == 1 {
            return Ok(x.clone());
        }
        if hp % m != 0 || wp % m != 0 {
            return Err(candle_core::Error::msg(format!(
                "spatial_merge: patch grid {}x{} not divisible by merge size {}",
                hp, wp, m
            )));
        }

        // (b, seq, hidden) -> (b, hp, wp, hidden)
        let x = x.contiguous()?.reshape((b, hp, wp, hidden))?;
        // (b, hp, wp, hidden) -> (b, hp/m, m, wp/m, m, hidden)
        let x = x.reshape((b, hp / m, m, wp / m, m, hidden))?;
        // (b, hp/m, wp/m, m, m, hidden)
        let x = x.transpose(2, 3)?.contiguous()?;
        // (b, (hp/m)*(wp/m), hidden*m*m)
        x.reshape((b, (hp / m) * (wp / m), hidden * m * m))
    }

    fn forward(&self, x: &Tensor, hp: usize, wp: usize) -> candle_core::Result<Tensor> {
        let x = self.norm.forward(x)?;
        let x = self.spatial_merge(&x, hp, wp)?;
        let x = self.fc1.forward(&x)?;
        let x = self.act.forward(&x)?;
        self.fc2.forward(&x)
    }
}

/// Vision encoder + merger producing language-hidden-size embeddings.
#[derive(Debug, Clone)]
pub struct VisionEncoder {
    cfg: VisionConfig,
    patch: Conv2d,
    pos_embed: Embedding,
    blocks: Vec<VitBlock>,
    merger: Merger,
    pos_cache: Arc<Mutex<HashMap<(usize, usize), Tensor>>>,
}

impl VisionEncoder {
    /// （新增）对外暴露必要的视觉配置参数，避免在上层直接访问私有字段。
    /// 为什么要加：图片缩放需要知道 patch_size / spatial_merge_size / num_position_embeddings 才能
    /// 按 Qwen3-VL 的 pipeline 做网格对齐与像素上下限约束。
    pub fn patch_size(&self) -> usize {
        self.cfg.patch_size
    }

    pub fn spatial_merge_size(&self) -> usize {
        self.cfg.spatial_merge_size
    }

    pub fn num_position_embeddings(&self) -> usize {
        self.cfg.num_position_embeddings
    }

    pub fn load(cfg: VisionConfig, vb: VarBuilder) -> Result<Self> {
        // Patch embed weights are stored as [out, in, temporal, kh, kw] for video support.
        // For images, we fold the temporal dimension into channels and run a 2D conv with
        // in_channels = in_channels * temporal_patch_size by duplicating the image frames.
        let patch_vb = vb.pp("model.visual.patch_embed.proj");
        let w5 = patch_vb.get(
            (
                cfg.hidden_size,
                cfg.in_channels,
                cfg.temporal_patch_size,
                cfg.patch_size,
                cfg.patch_size,
            ),
            "weight",
        )?;
        let b = patch_vb.get(cfg.hidden_size, "bias")?;
        let w = w5.reshape((
            cfg.hidden_size,
            cfg.in_channels * cfg.temporal_patch_size,
            cfg.patch_size,
            cfg.patch_size,
        ))?;

        let mut conv_cfg = Conv2dConfig::default();
        conv_cfg.stride = cfg.patch_size;
        let patch = Conv2d::new(w, Some(b), conv_cfg);

        let pos_embed = candle_nn::embedding(
            cfg.num_position_embeddings,
            cfg.hidden_size,
            vb.pp("model.visual.pos_embed"),
        )?;

        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(VitBlock::load(vb.pp(&format!("model.visual.blocks.{i}")), &cfg)?);
        }

        let merger = Merger::load(vb.pp("model.visual.merger"), cfg.hidden_size, cfg.spatial_merge_size, cfg.out_hidden_size)?;

        Ok(Self {
            cfg,
            patch,
            pos_embed,
            blocks,
            merger,
            pos_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    fn clamp_and_round_hw(&self, h: u32, w: u32) -> (usize, usize) {
        // Ensure patch count doesn't exceed num_position_embeddings.
        let max_side = 768u32; // 48*16 => 2304 positions max for square images.
        let mut nh = h;
        let mut nw = w;
        if nh.max(nw) > max_side {
            let scale = max_side as f32 / nh.max(nw) as f32;
            nh = (nh as f32 * scale) as u32;
            nw = (nw as f32 * scale) as u32;
        }
        // Round down to multiples of patch size.
        let ps = self.cfg.patch_size as u32;
        nh = (nh / ps).max(1) * ps;
        nw = (nw / ps).max(1) * ps;
        (nh as usize, nw as usize)
    }

    /// Convert an RGB image tensor in `[H,W,3]` (u8 or f32) into a model input tensor `[1,3,H,W]` (f32),
    /// applying `(x/255 - mean)/std` with mean/std 0.5.
    pub fn image_to_tensor(&self, rgb: &[u8], h: usize, w: usize, device: &Device) -> Result<Tensor> {
        if rgb.len() != h * w * 3 {
            bail!("invalid rgb buffer, expected {} bytes got {}", h * w * 3, rgb.len());
        }

        let mut data = Vec::with_capacity(h * w * 3);
        for v in rgb {
            let x = *v as f32 / 255.0;
            // mean=0.5 std=0.5
            data.push((x - 0.5) / 0.5);
        }
        let t = Tensor::from_vec(data, (h, w, 3), device)?;
        let t = t.transpose(0, 2)?.transpose(1, 2)?; // (3, h, w)
        Ok(t.unsqueeze(0)?) // (1, 3, h, w)
    }

    /// Encode an image input tensor `[1,3,H,W]` into projected features `[1,N, out_hidden_size]`.
    pub fn encode(&self, image: &Tensor) -> Result<Tensor> {
        let t0 = Instant::now();
        let (_b, _c, h, w) = image.dims4().map_err(|e| anyhow!("image dims4 -> {e}"))?;

        // Patch embedding.
        let t_patch = Instant::now();
        let x = if self.cfg.temporal_patch_size <= 1 {
            self.patch.forward(image)?
        } else {
            let t = self.cfg.temporal_patch_size;
            let mut frames: Vec<Tensor> = Vec::with_capacity(t);
            for _ in 0..t {
                frames.push(image.clone());
            }
            let frame_refs: Vec<&Tensor> = frames.iter().collect();
            let image = Tensor::cat(&frame_refs, 1)?; // (1, 3*T, h, w)
            self.patch.forward(&image)?
        }; // (1, hidden, h', w')

        let (_b, _hidden, hp0, wp0) = x.dims4()?;
        let patch_s = t_patch.elapsed().as_secs_f64();

        // Ensure the patch grid is compatible with spatial merging by cropping to multiples.
        let m = self.cfg.spatial_merge_size.max(1);
        if hp0 < m || wp0 < m {
            bail!(
                "image too small for spatial_merge_size {}: patch grid {}x{} (try larger image)",
                m,
                hp0,
                wp0
            );
        }
        let hp = (hp0 / m) * m;
        let wp = (wp0 / m) * m;
        let x = if hp != hp0 || wp != wp0 {
            x.narrow(2, 0, hp)?.narrow(3, 0, wp)?
        } else {
            x
        };
        let seq = hp * wp;

        if seq > self.cfg.num_position_embeddings {
            bail!(
                "too many visual tokens: {} ({}x{}), max is {} (try smaller image)",
                seq,
                hp,
                wp,
                self.cfg.num_position_embeddings
            );
        }

        let x = x.flatten_from(2)?.transpose(1, 2)?.contiguous()?; // (1, seq, hidden)

        // Add pos embeddings.
        let t_pos = Instant::now();
        //
        // 关键点：pos_embed 是按 48x48 网格展平存储的（num_position_embeddings=2304）。
        // 不能直接用 0..seq 这一段连续 id，否则当 patch grid 不是 48x48 时（比如 40x40、48x27）
        // 会把二维网格“错误地映射”到一维位置，导致视觉特征对不齐，模型输出容易出现乱码/噪声。
        let pos = {
            if let Ok(cache) = self.pos_cache.lock() {
                cache.get(&(hp, wp)).cloned()
            } else {
                None
            }
        };
        let pos = match pos {
            Some(pos) => pos,
            None => {
                let base_seq = self.cfg.num_position_embeddings;
                let base_side = (base_seq as f64).sqrt() as usize;
                if base_side * base_side != base_seq {
                    bail!("num_position_embeddings {} is not a square", base_seq);
                }
                if hp > base_side || wp > base_side {
                    bail!(
                        "patch grid {}x{} exceeds base {}x{} (resize image smaller)",
                        hp,
                        wp,
                        base_side,
                        base_side
                    );
                }
                let mut pos_id_vec: Vec<u32> = Vec::with_capacity(seq);
                for r in 0..hp {
                    for c in 0..wp {
                        pos_id_vec.push((r * base_side + c) as u32);
                    }
                }
                let pos_ids = Tensor::from_vec(pos_id_vec, seq, image.device())?;
                let pos = self.pos_embed.forward(&pos_ids)?.unsqueeze(0)?; // (1, seq, hidden)
                if let Ok(mut cache) = self.pos_cache.lock() {
                    cache.insert((hp, wp), pos.clone());
                }
                pos
            }
        };

        let mut x = (x + pos)?;
        let pos_s = t_pos.elapsed().as_secs_f64();

        // ViT blocks.
        let t_blocks = Instant::now();
        for blk in &self.blocks {
            x = blk.forward(&x)?;
        }
        let blocks_s = t_blocks.elapsed().as_secs_f64();

        // Project to language hidden size.
        let t_merger = Instant::now();
        let x = self.merger.forward(&x, hp, wp)?;
        let merger_s = t_merger.elapsed().as_secs_f64();

        let total_s = t0.elapsed().as_secs_f64();
        // 用 info 打印，方便定位 TTFT 慢到底卡在 patch/pos/blocks/merger 哪一段。
        log::info!(
            "vision encoded: input_hw={}x{} patch_grid={}x{} seq={} (patch_s={:.3} pos_s={:.3} blocks_s={:.3} merger_s={:.3} total_s={:.3})",
            h,
            w,
            hp,
            wp,
            seq
            ,patch_s
            ,pos_s
            ,blocks_s
            ,merger_s
            ,total_s
        );

        Ok(x)
    }
}
