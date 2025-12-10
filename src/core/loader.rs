use anyhow::{Context, Result};
use half::bf16;
use log::*;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::fs::File;
use std::path::Path;

use crate::core::config::LlamaConfig;
use crate::core::layer::{KVCache, Linear, LlamaAttention, LlamaBlock, LlamaMLP, LlamaRMSNorm};
use crate::core::model::LlamaModel;
use crate::core::shape::Shape;
use crate::core::tensor::Tensor;

pub struct Loader {
    mmap: memmap2::Mmap,
}

impl Loader {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        Ok(Self { mmap })
    }

    fn read_tensor(&self, tensors: &SafeTensors, name: &str) -> Result<(Vec<f32>, Shape)> {
        let view = tensors
            .tensor(name)
            .context(format!("Tensor not found: {}", name))?;
        let shape = Shape::new(view.shape().iter().map(|&x| x).collect());

        let data = match view.dtype() {
            safetensors::Dtype::F32 => view
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            safetensors::Dtype::BF16 => view
                .data()
                .chunks_exact(2)
                .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                .collect(),
            _ => anyhow::bail!("Unsupported dtype: {:?}", view.dtype()),
        };

        Ok((data, shape))
    }

    fn load_tensor(&self, tensors: &SafeTensors, name: &str) -> Result<Tensor> {
        let (data, shape) = self.read_tensor(tensors, name)?;
        debug!("Loading tensor: {} with shape {:?}", name, shape);
        let mut tensor = Tensor::new(data, shape);
        tensor.set_name(name.to_string());
        Ok(tensor)
    }

    fn load_gqa_tensor(
        &self,
        tensors: &SafeTensors,
        name: &str,
        config: &LlamaConfig,
    ) -> Result<Tensor> {
        let (src_data, shape) = self.read_tensor(tensors, name)?;

        let n_heads = config.num_heads;
        let n_kv_heads = config.num_kv_heads;

        if n_heads == n_kv_heads {
            return Ok(Tensor::new(src_data, shape));
        }

        // GQA Expansion
        let n_rep = n_heads / n_kv_heads;
        let dims = shape.dims();
        let out_dim = dims[0]; // [Out, In] assumption for weights
        let in_dim = dims[1];
        let head_dim = out_dim / n_kv_heads;

        let mut new_data = Vec::with_capacity(out_dim * n_rep * in_dim);
        let block_size = head_dim * in_dim;

        for i in 0..n_kv_heads {
            let start = i * block_size;
            let end = (i + 1) * block_size;
            let chunk = &src_data[start..end];
            for _ in 0..n_rep {
                new_data.extend_from_slice(chunk);
            }
        }

        let new_shape = Shape::new(vec![out_dim * n_rep, in_dim]);
        Ok(Tensor::new(new_data, new_shape))
    }

    pub fn load_model(&self, config: &LlamaConfig) -> Result<LlamaModel> {
        let tensors = SafeTensors::deserialize(&self.mmap)?;
        println!("Loading weights from safetensors...");

        // [치명적 버그 수정]
        // Embedding은 Lookup Table이므로 Transpose하면 안 됩니다!
        // [Vocab, Dim] 형태를 유지해야 main.rs에서 올바르게 슬라이싱 할 수 있습니다.
        let embed_tokens = self.load_tensor(&tensors, "model.embed_tokens.weight")?;

        let mut layers = Vec::new();
        for i in 0..config.num_layers {
            let p = format!("model.layers.{}", i);

            // Attention Weights: Linear 연산을 위해 Transpose 필요
            let q = self.load_tensor(&tensors, &format!("{}.self_attn.q_proj.weight", p))?;
            let k =
                self.load_gqa_tensor(&tensors, &format!("{}.self_attn.k_proj.weight", p), config)?;
            let v =
                self.load_gqa_tensor(&tensors, &format!("{}.self_attn.v_proj.weight", p), config)?;
            let o = self.load_tensor(&tensors, &format!("{}.self_attn.o_proj.weight", p))?;

            // MLP Weights
            let gate = self.load_tensor(&tensors, &format!("{}.mlp.gate_proj.weight", p))?;
            let up = self.load_tensor(&tensors, &format!("{}.mlp.up_proj.weight", p))?;
            let down = self.load_tensor(&tensors, &format!("{}.mlp.down_proj.weight", p))?;

            // Norms (1D)
            let input_norm =
                self.load_tensor(&tensors, &format!("{}.input_layernorm.weight", p))?;
            let post_norm =
                self.load_tensor(&tensors, &format!("{}.post_attention_layernorm.weight", p))?;

            let cache = KVCache::new(
                config.num_heads,
                config.hidden_size / config.num_heads,
                2048,
            ); // SeqLen 넉넉하게

            layers.push(LlamaBlock {
                attn: LlamaAttention::new(
                    Linear::new(q),
                    Linear::new(k),
                    Linear::new(v),
                    Linear::new(o),
                    config.num_heads,
                    cache,
                ),
                mlp: LlamaMLP::new(Linear::new(gate), Linear::new(up), Linear::new(down)),
                input_norm: LlamaRMSNorm::new(input_norm, config.rms_norm_eps),
                post_norm: LlamaRMSNorm::new(post_norm, config.rms_norm_eps),
            });
        }

        let norm = self.load_tensor(&tensors, "model.norm.weight")?;

        // 4. LM Head
        // LM Head는 Linear 연산(MatMul)을 하므로 [Dim, Vocab] 형태여야 합니다.
        // 따라서 파일에서 로드한 뒤 Transpose를 해야 합니다.
        let lm_head = if let Ok(w) = self.load_tensor(&tensors, "lm_head.weight") {
            w
        } else {
            println!("Weight Tying: Sharing embed_tokens with lm_head");
            // embed_tokens는 현재 [Vocab, Dim] (Row-Major) 입니다.
            // lm_head는 이를 전치하여 [Dim, Vocab]으로 만들어야 합니다.
            embed_tokens.clone()
        };

        Ok(LlamaModel {
            embed_tokens,
            layers,
            norm: LlamaRMSNorm::new(norm, config.rms_norm_eps),
            lm_head: Linear::new(lm_head),
        })
    }
}
