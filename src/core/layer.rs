use super::tensor::Tensor;
use crate::{core::shape::Shape, profile};
use log::*;
use rayon::prelude::*;
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
    pub gate_up_proj: Linear, // Gate와 Up이 합쳐진 레이어
    pub down_proj: Linear,
    pub intermediate_size: usize, // 반으로 쪼갤 위치
}
impl LlamaMLP {
    pub fn new(gate_up: Linear, down: Linear, intermediate_size: usize) -> Self {
        Self {
            gate_up_proj: gate_up,
            down_proj: down,
            intermediate_size,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // 1. Fused Projection (Gate + Up 한번에 계산)
        // 결과 Shape: [Seq, Intermediate * 2]
        let fused = self.gate_up_proj.forward(x);
        let src_data = fused.data(); // Read-only access

        let seq_len = x.shape().dims()[0];
        let mid = self.intermediate_size;

        // 차원 검증 (디버깅용)
        if src_data.len() != seq_len * mid * 2 {
            panic!(
                "MLP Size Mismatch: Seq={}, Inter={}, Expected={}, Got={}",
                seq_len,
                mid,
                seq_len * mid * 2,
                src_data.len()
            );
        }

        // 2. Fused SiLU + Mul & Packing (Zero-Copy Logic)
        // 결과를 저장할 버퍼를 미리 만듭니다 (크기: Seq * Intermediate)
        // 기존의 불필요한 Up 데이터 공간을 제거하고 압축하여 저장합니다.
        let mut result_data = vec![0.0; seq_len * mid];

        // 값이 클수록 빨라지지만 멍청해질 수 있음. 0.01 ~ 0.05 추천
        let threshold = 0.02;

        // Rayon을 이용해 병렬 처리
        // src(2*mid 크기)에서 읽어서 -> dst(mid 크기)에 씁니다.
        result_data
            .par_chunks_mut(mid)
            .zip(src_data.par_chunks(mid * 2))
            .for_each(|(dst, src)| {
                // src는 [Gate... (mid개), Up... (mid개)] 형태로 되어 있음
                let (gate, up) = src.split_at(mid);

                // 루프 최적화 (Auto-Vectorization 유도)
                for i in 0..mid {
                    let g = gate[i];
                    let u = up[i];
                    let silu = g / (1.0 + (-g).exp());
                    dst[i] = silu * u;

                    // [핵심] 값이 너무 작으면 0으로 죽임 (Sparsity 유도)
                    let val = silu * u;
                    if val.abs() < threshold {
                        dst[i] = 0.0;
                    } else {
                        dst[i] = val;
                    }
                }
            });

        // 3. Down Projection
        // 압축된 데이터를 바로 Tensor로 포장 (복사 없음)
        let input = Tensor::new(result_data, Shape::new(vec![seq_len, mid]));

        self.down_proj.forward(&input)
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

    fn extract_head_impl(tensor: &Tensor, head_idx: usize, n_heads: usize) -> Tensor {
        let dims = tensor.shape().dims();
        let seq_len = dims[0];
        let hidden = dims[1];
        let head_dim = hidden / n_heads;
        let mut new_data = Vec::with_capacity(seq_len * head_dim);
        let src = tensor.data();

        for i in 0..seq_len {
            let start = i * hidden + head_idx * head_dim;
            new_data.extend_from_slice(&src[start..start + head_dim]);
        }
        Tensor::new(new_data, Shape::new(vec![seq_len, head_dim]))
    }

    pub fn forward(&self, x: &Tensor, start_pos: usize) -> Tensor {
        // 1. Q, K, V Projection
        let q_all = self.q_proj.forward(x);
        let k_all = self.k_proj.forward(x);
        let v_all = self.v_proj.forward(x);

        let seq_len = x.shape().dims()[0];
        let hidden = x.shape().dims()[1];
        let n_heads = self.n_heads;
        let head_dim = hidden / n_heads;
        let max_seq = self.cache.max_seq_len;

        // 결과 저장용 버퍼
        let mut context_data = vec![0.0; seq_len * hidden];

        // KV Cache의 Raw Pointer 추출 (RefCell 우회)
        let mut k_cache_guard = self.cache.k.borrow_mut();
        let mut v_cache_guard = self.cache.v.borrow_mut();

        let k_base_addr = k_cache_guard.data_mut().as_mut_ptr() as usize;
        let v_base_addr = v_cache_guard.data_mut().as_mut_ptr() as usize;
        let ctx_base_addr = context_data.as_mut_ptr() as usize;

        // Read-Only 텐서는 참조 공유
        let q_ref = &q_all;
        let k_ref = &k_all;
        let v_ref = &v_all;

        // 2. Parallel Head Processing (Rayon)
        // [수정] move 키워드 추가: SendPtr를 복사해서 캡처하도록 강제
        (0..n_heads).into_par_iter().for_each(move |h| {
            // 포인터 복원
            let k_base_ptr = k_base_addr as *mut f32;
            let v_base_ptr = v_base_addr as *mut f32;
            let ctx_base_ptr = ctx_base_addr as *mut f32;

            // 정적 함수 호출
            let mut q_head = Self::extract_head_impl(q_ref, h, n_heads);
            let mut k_head = Self::extract_head_impl(k_ref, h, n_heads);
            let v_head = Self::extract_head_impl(v_ref, h, n_heads);

            q_head.apply_rope_inplace(start_pos);
            k_head.apply_rope_inplace(start_pos);

            // 1. KV Cache Update (Unsafe Pointer Arithmetic)
            unsafe {
                let k_src = k_head.data();
                let v_src = v_head.data();

                let k_head_offset = h * (max_seq * head_dim);
                let v_head_offset = h * (head_dim * max_seq);

                for i in 0..seq_len {
                    let pos = start_pos + i;
                    let src_idx = i * head_dim;

                    // K Update
                    let k_dest_idx = k_head_offset + pos * head_dim;
                    let k_dest_ptr = k_base_ptr.add(k_dest_idx);
                    std::ptr::copy_nonoverlapping(
                        k_src.as_ptr().add(src_idx),
                        k_dest_ptr,
                        head_dim,
                    );

                    // V Update (Transposed)
                    for d in 0..head_dim {
                        let v_dest_idx = v_head_offset + d * max_seq + pos;
                        *v_base_ptr.add(v_dest_idx) = v_src[src_idx + d];
                    }
                }
            }

            // 2. Attention Score
            let total_len = start_pos + seq_len;

            // Cache 읽기 (Unsafe Slice)
            let k_slice = unsafe {
                let k_start = h * (max_seq * head_dim);
                std::slice::from_raw_parts(k_base_ptr.add(k_start), total_len * head_dim)
            };

            let scores = q_head.matmul_slice(k_slice, total_len, head_dim);
            let mut scaled_scores = scores.scale(1.0 / (head_dim as f32).sqrt());

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

            // 3. Context Calculation
            let v_head_offset = h * (head_dim * max_seq);
            let probs_data = probs.data();

            if seq_len == 1 {
                // Decoding Optimization
                let mut ctx_vec = vec![0.0; head_dim];
                unsafe {
                    for d in 0..head_dim {
                        let v_row_start = v_head_offset + d * max_seq;
                        let v_row_slice =
                            std::slice::from_raw_parts(v_base_ptr.add(v_row_start), total_len);

                        let sum: f32 = probs_data
                            .iter()
                            .zip(v_row_slice.iter())
                            .map(|(a, b)| a * b)
                            .sum();
                        ctx_vec[d] = sum;
                    }

                    let dest = h * head_dim;
                    std::ptr::copy_nonoverlapping(
                        ctx_vec.as_ptr(),
                        ctx_base_ptr.add(dest),
                        head_dim,
                    );
                }
            } else {
                // Prefill Optimization
                let mut v_hist = Vec::with_capacity(total_len * head_dim);
                unsafe {
                    for i in 0..total_len {
                        for d in 0..head_dim {
                            let idx = v_head_offset + d * max_seq + i;
                            v_hist.push(*v_base_ptr.add(idx));
                        }
                    }
                }
                let v_view = Tensor::new(v_hist, Shape::new(vec![total_len, head_dim]));
                let context = probs.matmul(&v_view);
                let ctx_src = context.data();

                unsafe {
                    for i in 0..seq_len {
                        let dest = i * hidden + h * head_dim;
                        let src = i * head_dim;
                        std::ptr::copy_nonoverlapping(
                            ctx_src.as_ptr().add(src),
                            ctx_base_ptr.add(dest),
                            head_dim,
                        );
                    }
                }
            }
        });

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
        let h_norm: Tensor;
        {
            profile!("4.2.1. LlamaBlock Attention");
            h_norm = self.input_norm.forward(x);
        }
        let mut attn_out: Tensor;
        {
            profile!("4.2.2. LlamaBlock Attention");
            attn_out = self.attn.forward(&h_norm, start_pos);
        }
        {
            profile!("4.2.3. LlamaBlock Attention Residual");
            attn_out.add_assign(x);
        }
        let h = attn_out;

        let h_norm2: Tensor;
        {
            profile!("4.2.4. LlamaBlock MLP");
            h_norm2 = self.post_norm.forward(&h);
        }

        let mut mlp_out: Tensor;
        {
            profile!("4.2.5. LlamaBlock MLP");
            mlp_out = self.mlp.forward(&h_norm2);
        }
        {
            profile!("4.2.6. LlamaBlock MLP Residual");
            mlp_out.add_assign(&h);
        }
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
