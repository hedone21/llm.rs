use anyhow::{Context, Result};
use half::bf16;
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

    fn load_tensor(&self, tensors: &SafeTensors, name: &str, quantize: bool) -> Result<Tensor> {
        let (data, shape) = self.read_tensor(tensors, name)?;
        // debug!("Loading tensor: {} ...", name); // 로그 줄이기
        let mut tensor = Tensor::new(data, shape);
        tensor.set_name(name);

        if quantize {
            // [최적화] 즉시 Q8 포맷으로 변환하여 메모리 절약 및 가속 준비
            Ok(tensor.quantize_q4())
        } else {
            Ok(tensor)
        }
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
        println!("Loading weights (with Q8 Quantization)...");

        // Embedding은 보통 양자화하지 않음 (Lookup 속도 및 정밀도 유지)
        let embed_tokens = self.load_tensor(&tensors, "model.embed_tokens.weight", false)?;

        let mut layers = Vec::new();
        for i in 0..config.num_layers {
            let p = format!("model.layers.{}", i);

            // [핵심] Attention 및 MLP의 Linear 가중치들을 양자화(True)
            // Transpose 없이 로드 -> Q8 변환 -> backend에서 Hybrid 연산
            let q = self.load_tensor(&tensors, &format!("{}.self_attn.q_proj.weight", p), true)?;

            // GQA Tensor (K, V)도 양자화 가능하지만, GQA 로직 복잡성을 피하기 위해 일단 F32 유지
            // (성능을 더 높이려면 load_gqa_tensor 내부에서도 quantize 수행 필요)
            let k = self
                .load_gqa_tensor(&tensors, &format!("{}.self_attn.k_proj.weight", p), config)?
                .quantize_q4();
            let v = self
                .load_gqa_tensor(&tensors, &format!("{}.self_attn.v_proj.weight", p), config)?
                .quantize_q4();

            let o = self.load_tensor(&tensors, &format!("{}.self_attn.o_proj.weight", p), true)?;

            // MLP 가중치 양자화 (가장 파라미터가 많음 -> 효과 큼)
            // 1. Raw Data(F32) 상태로 읽기
            let (gate_data, gate_shape) =
                self.read_tensor(&tensors, &format!("{}.mlp.gate_proj.weight", p))?;
            let (up_data, _up_shape) =
                self.read_tensor(&tensors, &format!("{}.mlp.up_proj.weight", p))?;

            // 2. 데이터 합치기: [Gate Rows] + [Up Rows]
            // Gate Shape: [Inter, Hidden], Up Shape: [Inter, Hidden]
            let inter_size = gate_shape.dims()[0];
            let hidden_size = gate_shape.dims()[1];

            let mut fused_data = gate_data; // Move gate_data
            fused_data.extend(up_data); // Append up_data

            // 3. 합쳐진 텐서 생성 및 양자화
            let fused_shape = Shape::new(vec![inter_size * 2, hidden_size]);
            let mut fused_tensor = Tensor::new(fused_data, fused_shape);
            fused_tensor.set_name(&format!("{}.mlp.gate_up_fused", p));

            // 4. Linear 레이어 생성 (Q4 양자화 적용)
            let gate_up_proj = Linear::new(fused_tensor.quantize_q4());

            // Down Proj는 기존대로 로드
            let down = self.load_tensor(&tensors, &format!("{}.mlp.down_proj.weight", p), true)?;

            // Norm은 파라미터가 적으므로 F32 유지 (정밀도 중요)
            let input_norm =
                self.load_tensor(&tensors, &format!("{}.input_layernorm.weight", p), false)?;
            let post_norm = self.load_tensor(
                &tensors,
                &format!("{}.post_attention_layernorm.weight", p),
                false,
            )?;

            // ... (KVCache 생성 등 기존 코드)
            let cache = KVCache::new(
                config.num_heads,
                config.hidden_size / config.num_heads,
                2048,
            );

            layers.push(LlamaBlock {
                attn: LlamaAttention::new(
                    Linear::new(q),
                    Linear::new(k),
                    Linear::new(v),
                    Linear::new(o),
                    config.num_heads,
                    cache,
                ),
                mlp: LlamaMLP::new(gate_up_proj, Linear::new(down), config.intermediate_size),
                input_norm: LlamaRMSNorm::new(input_norm, config.rms_norm_eps),
                post_norm: LlamaRMSNorm::new(post_norm, config.rms_norm_eps),
            });
        }

        let norm = self.load_tensor(&tensors, "model.norm.weight", false)?;

        // LM Head도 크기가 크므로 양자화 추천 (단, 정확도 민감하면 false)
        let lm_head = if let Ok(w) = self.load_tensor(&tensors, "lm_head.weight", true) {
            w
        } else {
            embed_tokens.clone().quantize_q4()
        };

        Ok(LlamaModel {
            embed_tokens,
            layers,
            norm: LlamaRMSNorm::new(norm, config.rms_norm_eps),
            lm_head: Linear::new(lm_head),
        })
    }
}
