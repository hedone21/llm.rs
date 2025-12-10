use super::tensor::Tensor;
use crate::core::shape::Shape;
use log::*;
use std::cell::RefCell;

pub struct Linear {
    pub weight: Tensor,
}
impl Linear {
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.matmul_transposed(&self.weight)
    }
}

pub struct LlamaMLP {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}
impl LlamaMLP {
    pub fn new(gate: Linear, up: Linear, down: Linear) -> Self {
        Self {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
        }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        gate.silu_mul_inplace(&up);
        self.down_proj.forward(&gate)
    }
}

pub struct KVCache {
    pub k: RefCell<Tensor>,
    pub v: RefCell<Tensor>,
}
impl KVCache {
    pub fn new(n_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let hidden = n_heads * head_dim;
        let shape = Shape::new(vec![max_seq_len, hidden]);
        Self {
            k: RefCell::new(Tensor::zeros(shape.clone())),
            v: RefCell::new(Tensor::zeros(shape)),
        }
    }
}

pub struct LlamaAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub n_heads: usize,
    pub cache: KVCache,
}

impl LlamaAttention {
    pub fn new(q: Linear, k: Linear, v: Linear, o: Linear, n_heads: usize, cache: KVCache) -> Self {
        Self {
            q_proj: q,
            k_proj: k,
            v_proj: v,
            o_proj: o,
            n_heads,
            cache,
        }
    }

    fn extract_head(&self, tensor: &Tensor, head_idx: usize) -> Tensor {
        let dims = tensor.shape().dims();
        let seq_len = dims[0];
        let hidden = dims[1];
        let head_dim = hidden / self.n_heads;
        let mut new_data = Vec::with_capacity(seq_len * head_dim);
        let src = tensor.data();

        for i in 0..seq_len {
            let start = i * hidden + head_idx * head_dim;
            new_data.extend_from_slice(&src[start..start + head_dim]);
        }
        Tensor::new(new_data, Shape::new(vec![seq_len, head_dim]))
    }

    pub fn forward(&self, x: &Tensor, start_pos: usize) -> Tensor {
        let q_all = self.q_proj.forward(x);
        let k_all = self.k_proj.forward(x);
        let v_all = self.v_proj.forward(x);

        let seq_len = x.shape().dims()[0];
        let hidden = x.shape().dims()[1];
        let head_dim = hidden / self.n_heads;

        let mut context_data = vec![0.0; seq_len * hidden];

        for h in 0..self.n_heads {
            let mut q_head = self.extract_head(&q_all, h);
            let k_head = self.extract_head(&k_all, h);
            let v_head = self.extract_head(&v_all, h);

            q_head.apply_rope_inplace(start_pos);

            let mut k_rot = k_head.clone();
            k_rot.apply_rope_inplace(start_pos);

            // KV Cache Update
            {
                let mut c_k = self.cache.k.borrow_mut();
                let mut c_v = self.cache.v.borrow_mut();
                let k_ptr = c_k.data_mut();
                let v_ptr = c_v.data_mut();

                let k_rot_data = k_rot.data();
                let v_head_data = v_head.data();

                for i in 0..seq_len {
                    let dest_idx = (start_pos + i) * hidden + h * head_dim;
                    let src_idx = i * head_dim;
                    k_ptr[dest_idx..dest_idx + head_dim]
                        .copy_from_slice(&k_rot_data[src_idx..src_idx + head_dim]);
                    v_ptr[dest_idx..dest_idx + head_dim]
                        .copy_from_slice(&v_head_data[src_idx..src_idx + head_dim]);
                }
            }

            // Attention
            let total_len = start_pos + seq_len;
            let c_k = self.cache.k.borrow();
            let c_v = self.cache.v.borrow();

            let mut k_hist = Vec::with_capacity(total_len * head_dim);
            let mut v_hist = Vec::with_capacity(total_len * head_dim);

            let c_k_data = c_k.data();
            let c_v_data = c_v.data();
            for i in 0..total_len {
                let idx = i * hidden + h * head_dim;
                k_hist.extend_from_slice(&c_k_data[idx..idx + head_dim]);
                v_hist.extend_from_slice(&c_v_data[idx..idx + head_dim]);
            }

            let k_view = Tensor::new(k_hist, Shape::new(vec![total_len, head_dim]));
            let v_view = Tensor::new(v_hist, Shape::new(vec![total_len, head_dim]));

            // [수정] Score = Q @ K^T
            // matmul_transposed는 A @ B^T를 계산하므로, K(TotalLen, HD)를 그대로 넘깁니다.
            // Q(Seq, HD) @ K(TL, HD)^T -> [Seq, TL]
            let scores = q_head.matmul_transposed(&k_view);

            let scaled_scores = scores.scale(1.0 / (head_dim as f32).sqrt());

            let mut masked_scores = scaled_scores;
            if seq_len > 1 {
                let data = masked_scores.data_mut();
                for i in 0..seq_len {
                    for j in 0..total_len {
                        if j > start_pos + i {
                            data[i * total_len + j] = f32::NEG_INFINITY;
                        }
                    }
                }
            }

            let probs = masked_scores.softmax();

            // [수정] Context = Probs @ V
            // Probs(Seq, TL) @ V(TL, HD).
            // 최적화를 위해 V를 전치하여 matmul_transposed 사용: Probs @ (V^T)^T
            let context = probs.matmul_transposed(&v_view.transpose());

            let ctx_ptr = context.data();
            for i in 0..seq_len {
                let dest = i * hidden + h * head_dim;
                let src = i * head_dim;
                context_data[dest..dest + head_dim].copy_from_slice(&ctx_ptr[src..src + head_dim]);
            }
        }

        self.o_proj.forward(&Tensor::new(
            context_data,
            Shape::new(vec![seq_len, hidden]),
        ))
    }
}

pub struct LlamaRMSNorm {
    pub weight: Tensor,
    pub eps: f32,
}
impl LlamaRMSNorm {
    pub fn new(weight: Tensor, eps: f32) -> Self {
        Self { weight, eps }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.rms_norm(&self.weight, self.eps)
    }
}

pub struct LlamaBlock {
    pub attn: LlamaAttention,
    pub mlp: LlamaMLP,
    pub input_norm: LlamaRMSNorm,
    pub post_norm: LlamaRMSNorm,
}

impl LlamaBlock {
    pub fn forward(&self, x: &Tensor, start_pos: usize) -> Tensor {
        let h_norm = self.input_norm.forward(x);
        let mut attn_out = self.attn.forward(&h_norm, start_pos);
        attn_out.add_assign(x);
        let h = attn_out;

        let h_norm2 = self.post_norm.forward(&h);
        let mut mlp_out = self.mlp.forward(&h_norm2);
        mlp_out.add_assign(&h);
        mlp_out
    }
}
// (Test 코드는 파일 끝에 이전과 같이 포함되어야 합니다)
#[cfg(test)]
mod tests {
    use super::*;
    // (이전 답변의 테스트 코드들: mock_linear, test_attention_causality, test_kv_cache_update 등)
    // mock_linear 함수는 반드시 여기에 다시 포함시켜야 컴파일됩니다.
    fn mock_linear(dim: usize) -> Linear {
        let mut data = vec![0.0; dim * dim];
        for i in 0..dim {
            data[i * dim + i] = 1.0;
        }
        Linear::new(Tensor::new(data, Shape::new(vec![dim, dim])))
    }

    #[test]
    fn test_attention_causality() {
        let dim = 4;
        let n_heads = 2;
        let attn = LlamaAttention::new(
            mock_linear(dim),
            mock_linear(dim),
            mock_linear(dim),
            mock_linear(dim),
            n_heads,
            KVCache::new(n_heads, dim / n_heads, 100),
        );
        let input1 = Tensor::new(
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            Shape::new(vec![2, 4]),
        );
        let out1 = attn.forward(&input1, 0);
        let out1_token0 = out1.data()[0..4].to_vec();

        let attn2 = LlamaAttention::new(
            mock_linear(dim),
            mock_linear(dim),
            mock_linear(dim),
            mock_linear(dim),
            n_heads,
            KVCache::new(n_heads, dim / n_heads, 100),
        );
        let input2 = Tensor::new(
            vec![1.0, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0],
            Shape::new(vec![2, 4]),
        );
        let out2 = attn2.forward(&input2, 0);
        let out2_token0 = out2.data()[0..4].to_vec();
        assert_eq!(out1_token0, out2_token0);
    }

    #[test]
    fn test_kv_cache_update() {
        let dim = 2;
        let n_heads = 1;
        let attn = LlamaAttention::new(
            mock_linear(dim),
            mock_linear(dim),
            mock_linear(dim),
            mock_linear(dim),
            n_heads,
            KVCache::new(n_heads, dim, 10),
        );
        let input = Tensor::new(vec![1.0, 2.0], Shape::new(vec![1, 2]));
        attn.forward(&input, 0);
        {
            let k_cache = attn.cache.k.borrow();
            assert_ne!(k_cache.data()[0], 0.0);
        }
        attn.forward(&input, 1);
        {
            let k_cache = attn.cache.k.borrow();
            assert_ne!(k_cache.data()[2], 0.0);
        }
    }
}
