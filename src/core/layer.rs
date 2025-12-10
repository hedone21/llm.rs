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
    pub max_seq_len: usize,
    pub head_dim: usize,
}
impl KVCache {
    pub fn new(n_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        // [Layout Optimization]
        // K: [Heads, MaxSeq, HeadDim] -> Head별로 연속적이라 읽기 편함
        // V: [Heads, HeadDim, MaxSeq] -> Transposed 저장! (Probs @ V 고속화를 위해)

        let k_shape = Shape::new(vec![n_heads, max_seq_len, head_dim]);
        let v_shape = Shape::new(vec![n_heads, head_dim, max_seq_len]);

        Self {
            k: RefCell::new(Tensor::zeros(k_shape)),
            v: RefCell::new(Tensor::zeros(v_shape)),
            max_seq_len,
            head_dim,
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

        // 이 부분은 어쩔 수 없이 strided copy가 발생하지만, seq_len이 작을 땐 빠릅니다.
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
        let max_seq = self.cache.max_seq_len;

        let mut context_data = vec![0.0; seq_len * hidden];

        for h in 0..self.n_heads {
            let mut q_head = self.extract_head(&q_all, h);
            let mut k_head = self.extract_head(&k_all, h);
            let v_head = self.extract_head(&v_all, h);

            q_head.apply_rope_inplace(start_pos);
            k_head.apply_rope_inplace(start_pos);

            // 1. KV Cache Update (Layout Optimized)
            {
                let mut c_k = self.cache.k.borrow_mut();
                let mut c_v = self.cache.v.borrow_mut();
                let k_ptr = c_k.data_mut();
                let v_ptr = c_v.data_mut();

                let k_src = k_head.data();
                let v_src = v_head.data();

                // Head Offset 계산
                let k_head_offset = h * (max_seq * head_dim);
                let v_head_offset = h * (head_dim * max_seq);

                for i in 0..seq_len {
                    let pos = start_pos + i;

                    // K Update: [Head, Pos, :] -> 연속 복사 (빠름)
                    let k_dest_idx = k_head_offset + pos * head_dim;
                    let src_idx = i * head_dim;
                    k_ptr[k_dest_idx..k_dest_idx + head_dim]
                        .copy_from_slice(&k_src[src_idx..src_idx + head_dim]);

                    // V Update: [Head, :, Pos] -> 전치 저장 (Transposed Storage)
                    // 나중에 Probs @ V 할 때 열(Column) 단위 접근을 피하기 위해 미리 돌려놓습니다.
                    for d in 0..head_dim {
                        let v_dest_idx = v_head_offset + d * max_seq + pos;
                        v_ptr[v_dest_idx] = v_src[src_idx + d];
                    }
                }
            }

            // 2. Attention Score (Zero-Copy)
            let total_len = start_pos + seq_len;
            let c_k = self.cache.k.borrow();

            // K Slice 추출 (복사 없음!)
            let k_start = h * (max_seq * head_dim);
            // 현재까지 유효한 데이터만 슬라이싱
            let k_slice = &c_k.data()[k_start..k_start + total_len * head_dim];

            // Q @ K^T
            // K_slice는 [TotalLen, HeadDim] 형태의 연속 메모리입니다.
            let scores = q_head.matmul_slice(k_slice, total_len, head_dim);

            let mut scaled_scores = scores.scale(1.0 / (head_dim as f32).sqrt());

            // Causal Masking
            if seq_len > 1 {
                let data = scaled_scores.data_mut();
                for i in 0..seq_len {
                    for j in 0..total_len {
                        if j > start_pos + i {
                            data[i * total_len + j] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            let probs = scaled_scores.softmax();

            // 3. Context Calculation (Zero-Copy & Optimized Decoding)
            // Context = Probs @ V
            // V는 [Head, HeadDim, MaxSeq] 형태로 저장되어 있습니다.
            // 즉, V의 각 행(Row)은 특정 차원(d)의 전체 시계열 데이터입니다.

            let c_v = self.cache.v.borrow();
            let v_data = c_v.data();
            let v_head_offset = h * (head_dim * max_seq);
            let probs_data = probs.data(); // [Seq, TotalLen]

            // Decoding 단계 (Seq=1) 최적화
            if seq_len == 1 {
                let mut ctx_vec = vec![0.0; head_dim];
                for d in 0..head_dim {
                    // V의 d번째 행 (길이 MaxSeq) 중 유효한 TotalLen만큼 슬라이싱
                    let v_row_start = v_head_offset + d * max_seq;
                    let v_row_slice = &v_data[v_row_start..v_row_start + total_len];

                    // 내적: Probs(1, TL) . V_row_d(TL)
                    // V가 전치되어 저장된 덕분에 연속 메모리 내적이 가능합니다.
                    let sum: f32 = probs_data
                        .iter()
                        .zip(v_row_slice.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    ctx_vec[d] = sum;
                }

                let dest = h * head_dim;
                context_data[dest..dest + head_dim].copy_from_slice(&ctx_vec);
            } else {
                // Prefill 단계 (Seq > 1)
                // 복잡한 인덱싱 대신 안전하게 Tensor로 재구성하여 연산 (초기 1회만 수행되므로 OK)
                let mut v_hist = Vec::with_capacity(total_len * head_dim);
                for i in 0..total_len {
                    for d in 0..head_dim {
                        let idx = v_head_offset + d * max_seq + i;
                        v_hist.push(v_data[idx]);
                    }
                }
                let v_view = Tensor::new(v_hist, Shape::new(vec![total_len, head_dim]));
                // 표준 연산: Probs(Seq, TL) @ V(TL, HD)
                let context = probs.matmul(&v_view);

                let ctx_ptr = context.data();
                for i in 0..seq_len {
                    let dest = i * hidden + h * head_dim;
                    let src = i * head_dim;
                    context_data[dest..dest + head_dim]
                        .copy_from_slice(&ctx_ptr[src..src + head_dim]);
                }
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
