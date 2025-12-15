use std::sync::Arc;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::{
    backend::{Backend, Device},
    core::{
        shape::Shape,
        tensor::{Storage, Tensor},
    },
};
use rayon::prelude::*;

pub struct CpuBackend;

impl Backend for CpuBackend {
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn name(&self) -> &str {
        "Hybrid SIMD (AVX2/NEON)"
    }

    // -------------------------------------------------------------------------
    // 2. MATMUL Transposed (Main Kernel)
    // -------------------------------------------------------------------------
    fn matmul_transposed(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let dims_a = a.shape().dims();
        let dims_b_t = b.shape().dims();
        let (m, k) = (dims_a[0], dims_a[1]);
        let (n, k2) = (dims_b_t[0], dims_b_t[1]);

        if k != k2 {
            panic!(
                "MatMul Transposed dimensions mismatch: A[{},{}] vs B[{},{}]",
                m, k, n, k2
            );
        }

        let mut result_data = vec![0.0; m * n];

        let a_data = match &*a.storage {
            Storage::Cpu(d) => d,
            _ => panic!("Input A must be F32"),
        };

        match &*b.storage {
            // [Case 1] Q4 Quantized Weights
            Storage::CpuQ4 {
                data: b_q,
                scales: b_s,
            } => {
                let block_size = 32;
                let num_blocks = k / block_size;

                // (A) Decoding / Batch=1 (Optimized Path)
                if m == 1 {
                    let mut a_q8 = vec![0i8; k];
                    let mut a_scales = vec![0.0f32; num_blocks];
                    self.quantize_row_q8_0(a_data, &mut a_q8, &mut a_scales);

                    let chunk_size = 256;
                    result_data.par_chunks_mut(chunk_size).enumerate().for_each(
                        |(cid, res_chunk)| {
                            let start_col = cid * chunk_size;
                            self.call_simd_kernel(
                                res_chunk, start_col, k, num_blocks, b_q, b_s, &a_q8, &a_scales,
                            );
                        },
                    );
                }
                // (B) Prefill / Batch > 1 (Row-wise Parallel)
                else {
                    result_data
                        .par_chunks_mut(n)
                        .enumerate()
                        .for_each(|(i, res_row)| {
                            // Quantize Activation Row
                            let a_row_start = i * k;
                            let a_row = &a_data[a_row_start..a_row_start + k];

                            let mut a_q8 = vec![0i8; k];
                            let mut a_scales = vec![0.0f32; num_blocks];
                            self.quantize_row_q8_0(a_row, &mut a_q8, &mut a_scales);

                            // Process Columns in chunks
                            let chunk_size = 256;
                            for (cid, res_chunk) in res_row.chunks_mut(chunk_size).enumerate() {
                                let start_col = cid * chunk_size;
                                self.call_simd_kernel(
                                    res_chunk, start_col, k, num_blocks, b_q, b_s, &a_q8, &a_scales,
                                );
                            }
                        });
                }
            }

            // [Case 2] F32 Weights (No Quantization)
            Storage::Cpu(b_data) => {
                if m == 1 {
                    result_data.par_iter_mut().enumerate().for_each(|(j, res)| {
                        let a_row = &a_data[0..k];
                        let b_row_start = j * k;
                        let b_row = &b_data[b_row_start..b_row_start + k];
                        let sum: f32 = a_row.iter().zip(b_row.iter()).map(|(x, y)| x * y).sum();
                        *res = sum;
                    });
                } else {
                    result_data
                        .par_chunks_mut(n)
                        .enumerate()
                        .for_each(|(i, res_row)| {
                            let a_row_start = i * k;
                            let a_row = &a_data[a_row_start..a_row_start + k];
                            for (j, res) in res_row.iter_mut().enumerate() {
                                let b_row_start = j * k;
                                let b_row = &b_data[b_row_start..b_row_start + k];
                                let sum: f32 =
                                    a_row.iter().zip(b_row.iter()).map(|(x, y)| x * y).sum();
                                *res = sum;
                            }
                        });
                }
            }
            _ => panic!("Unsupported storage type for B in matmul_transposed"),
        }

        Tensor::new(result_data, Shape::new(vec![m, n]))
    }

    // [기본 행렬 곱] A @ B
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let dims_a = a.shape().dims();
        let dims_b = b.shape().dims();

        let (m, k) = (dims_a[0], dims_a[1]);
        let (k2, n) = (dims_b[0], dims_b[1]);

        if k != k2 {
            panic!(
                "MatMul Dimension mismatch: ({}, {}) @ ({}, {})",
                m, k, k2, n
            );
        }

        let mut result_data = vec![0.0; m * n];

        let a_data = match &*a.storage {
            Storage::Cpu(d) => d,
            _ => panic!("CpuBackend requires CPU storage for input A"),
        };
        let b_data = match &*b.storage {
            Storage::Cpu(d) => d,
            _ => panic!("CpuBackend requires CPU storage for input B"),
        };

        if m == 1 {
            result_data.par_iter_mut().enumerate().for_each(|(j, res)| {
                let mut sum = 0.0;
                for p in 0..k {
                    unsafe {
                        sum += *a_data.get_unchecked(p) * *b_data.get_unchecked(p * n + j);
                    }
                }
                *res = sum;
            });
        } else {
            result_data
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, res_row)| {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for p in 0..k {
                            unsafe {
                                sum += *a_data.get_unchecked(i * k + p)
                                    * *b_data.get_unchecked(p * n + j);
                            }
                        }
                        res_row[j] = sum;
                    }
                });
        }

        Tensor::new(result_data, Shape::new(vec![m, n]))
    }

    // [Zero-Copy 슬라이스 연산] A @ Slice^T
    fn matmul_slice(&self, a: &Tensor, other_data: &[f32], rows: usize, cols: usize) -> Tensor {
        let dims_a = a.shape().dims();
        let (m, k) = (dims_a[0], dims_a[1]);

        if k != cols {
            panic!(
                "MatMul Slice mismatch: A[{},{}] vs Slice[{},{}]",
                m, k, rows, cols
            );
        }

        let mut result_data = vec![0.0; m * rows];
        let a_data = match &*a.storage {
            Storage::Cpu(d) => d,
            _ => panic!("CpuBackend requires CPU storage"),
        };

        if m == 1 {
            result_data.par_iter_mut().enumerate().for_each(|(j, res)| {
                let a_slice = &a_data[0..k];
                let b_row_start = j * cols;
                let b_slice = &other_data[b_row_start..b_row_start + cols];

                let sum: f32 = a_slice.iter().zip(b_slice.iter()).map(|(x, y)| x * y).sum();
                *res = sum;
            });
        } else {
            result_data
                .par_chunks_mut(rows)
                .enumerate()
                .for_each(|(i, res_row)| {
                    let a_row_start = i * k;
                    let a_slice = &a_data[a_row_start..a_row_start + k];

                    for j in 0..rows {
                        let b_row_start = j * cols;
                        let b_slice = &other_data[b_row_start..b_row_start + cols];
                        let sum: f32 = a_slice.iter().zip(b_slice.iter()).map(|(x, y)| x * y).sum();
                        res_row[j] = sum;
                    }
                });
        }

        Tensor::new(result_data, Shape::new(vec![m, rows]))
    }

    fn add_assign(&self, a: &mut Tensor, b: &Tensor) {
        if a.shape() != b.shape() {
            panic!("Shape mismatch in add_assign");
        }
        let a_data = match Arc::get_mut(&mut a.storage) {
            Some(Storage::Cpu(d)) => d,
            _ => panic!("Cannot mutate shared/non-cpu storage"),
        };
        let b_data = match &*b.storage {
            Storage::Cpu(d) => d,
            _ => panic!("CpuBackend requires CPU storage"),
        };
        a_data
            .par_iter_mut()
            .zip(b_data.par_iter())
            .for_each(|(x, y)| *x += *y);
    }

    fn silu_mul(&self, gate: &mut Tensor, up: &Tensor) {
        if gate.shape() != up.shape() {
            panic!("Shape mismatch in silu_mul");
        }
        let gate_data = match Arc::get_mut(&mut gate.storage) {
            Some(Storage::Cpu(d)) => d,
            _ => panic!("Cannot mutate shared/non-cpu storage"),
        };
        let up_data = match &*up.storage {
            Storage::Cpu(d) => d,
            _ => panic!("CpuBackend requires CPU storage"),
        };
        gate_data
            .par_iter_mut()
            .zip(up_data.par_iter())
            .for_each(|(g, u)| {
                let val = *g;
                let silu = val / (1.0 + (-val).exp());
                *g = silu * *u;
            });
    }

    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize) {
        let dims = x.shape().dims();
        let head_dim = *dims.last().unwrap();
        let mid = head_dim / 2;
        let theta_base = 500_000.0f32;

        let x_data = match Arc::get_mut(&mut x.storage) {
            Some(Storage::Cpu(d)) => d,
            _ => panic!("Cannot mutate shared/non-cpu storage"),
        };

        x_data
            .par_chunks_mut(head_dim)
            .enumerate()
            .for_each(|(i, chunk)| {
                let pos = start_pos + i;
                for j in 0..mid {
                    let freq_idx = (j as f32 * 2.0) / (head_dim as f32);
                    let theta = 1.0 / theta_base.powf(freq_idx);
                    let m_theta = (pos as f32) * theta;
                    let (sin, cos) = m_theta.sin_cos();

                    let re = chunk[j];
                    let im = chunk[j + mid];

                    chunk[j] = re * cos - im * sin;
                    chunk[j + mid] = re * sin + im * cos;
                }
            });
    }

    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
        let dims = x.shape().dims();
        let last_dim = *dims.last().unwrap();

        let x_data = match &*x.storage {
            Storage::Cpu(d) => d,
            _ => panic!("CpuBackend requires CPU storage"),
        };
        let w_data = match &*weight.storage {
            Storage::Cpu(d) => d,
            _ => panic!("CpuBackend requires CPU storage"),
        };

        assert_eq!(
            w_data.len(),
            last_dim,
            "Weight dimension mismatch in RMS Norm"
        );

        let mut output_data = vec![0.0; x_data.len()];

        output_data
            .par_chunks_mut(last_dim)
            .zip(x_data.par_chunks(last_dim))
            .for_each(|(out_row, in_row)| {
                let ss: f32 = in_row.iter().fold(0.0, |acc, &v| acc + v * v);
                let rms = (ss / last_dim as f32 + eps).sqrt();
                let scale = 1.0 / rms;

                out_row
                    .iter_mut()
                    .zip(in_row.iter())
                    .zip(w_data.iter())
                    .for_each(|((out, &x_val), &w_val)| {
                        *out = x_val * scale * w_val;
                    });
            });

        Tensor::new(output_data, x.shape().clone())
    }

    fn softmax(&self, x: &Tensor) -> Tensor {
        let dims = x.shape().dims();
        let last_dim = *dims.last().unwrap();

        let x_data = match &*x.storage {
            Storage::Cpu(d) => d,
            _ => panic!("CpuBackend requires CPU storage"),
        };

        let mut output_data = vec![0.0; x_data.len()];

        output_data
            .par_chunks_mut(last_dim)
            .zip(x_data.par_chunks(last_dim))
            .for_each(|(out_row, in_row)| {
                let max_val = in_row.iter().fold(f32::MIN, |a, &b| a.max(b));
                let mut sum_exp = 0.0;
                for j in 0..last_dim {
                    let e = (in_row[j] - max_val).exp();
                    out_row[j] = e;
                    sum_exp += e;
                }
                let inv_sum = 1.0 / sum_exp;
                for j in 0..last_dim {
                    out_row[j] *= inv_sum;
                }
            });

        Tensor::new(output_data, x.shape().clone())
    }

    fn scale(&self, x: &Tensor, value: f32) -> Tensor {
        let x_data = match &*x.storage {
            Storage::Cpu(d) => d,
            _ => panic!("CpuBackend requires CPU storage"),
        };
        let new_data: Vec<f32> = x_data.par_iter().map(|&e| e * value).collect();
        Tensor::new(new_data, x.shape().clone())
    }

    fn copy_from(&self, tensor: &Tensor) -> Tensor {
        match &*tensor.storage {
            Storage::Cpu(data) => Tensor::new(data.clone(), tensor.shape().clone()),
            _ => panic!("Copying from non-CPU storage not implemented yet"),
        }
    }
}

impl CpuBackend {
    // -------------------------------------------------------------------------
    // Helper: SIMD Dispatcher
    // -------------------------------------------------------------------------
    #[inline(always)]
    fn call_simd_kernel(
        &self,
        res_chunk: &mut [f32],
        start_col: usize,
        k: usize,
        num_blocks: usize,
        b_q: &[u8],
        b_s: &[f32],
        a_q8: &[i8],
        a_scales: &[f32],
    ) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self::kernel_neon_sdot(
                res_chunk, start_col, k, num_blocks, b_q, b_s, a_q8, a_scales,
            );
        }

        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self::kernel_avx2(
                res_chunk, start_col, k, num_blocks, b_q, b_s, a_q8, a_scales,
            );
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Fallback
            for x in res_chunk.iter_mut() {
                *x = 0.0;
            }
        }
    }

    // -------------------------------------------------------------------------
    // 1. Activation Quantization (F32 -> Q8)
    // -------------------------------------------------------------------------
    fn quantize_row_q8_0(&self, data: &[f32], out_q: &mut [i8], out_s: &mut [f32]) {
        let block_size = 32;

        out_q
            .par_chunks_mut(block_size)
            .zip(out_s.par_iter_mut())
            .zip(data.par_chunks(block_size))
            .for_each(|((q_blk, s_blk), src)| {
                unsafe {
                    #[cfg(target_arch = "aarch64")]
                    {
                        // ARM NEON Implementation
                        let mut max_v = vdupq_n_f32(0.0);
                        for i in 0..8 {
                            let val = vld1q_f32(src.as_ptr().add(i * 4));
                            max_v = vmaxnmq_f32(max_v, vabsq_f32(val));
                        }
                        let max_abs = vmaxnmvq_f32(max_v);

                        if max_abs == 0.0 {
                            *s_blk = 0.0;
                            // q_blk fill 0
                            let ptr = q_blk.as_mut_ptr() as *mut u8;
                            let zero = vdupq_n_u8(0);
                            vst1q_u8(ptr, zero);
                            vst1q_u8(ptr.add(16), zero);
                        } else {
                            let scale = max_abs / 127.0;
                            let inv_scale = 1.0 / scale;
                            *s_blk = scale;

                            let v_inv = vdupq_n_f32(inv_scale);
                            let q_ptr = q_blk.as_mut_ptr();
                            for i in 0..8 {
                                let val = vld1q_f32(src.as_ptr().add(i * 4));
                                let scaled = vmulq_f32(val, v_inv);
                                let rounded = vcvtaq_s32_f32(scaled);
                                let narrowed = vmovn_s32(rounded);

                                // Compact fallback for i16 -> i8 extraction
                                let v0 = vget_lane_s16(narrowed, 0) as i8;
                                let v1 = vget_lane_s16(narrowed, 1) as i8;
                                let v2 = vget_lane_s16(narrowed, 2) as i8;
                                let v3 = vget_lane_s16(narrowed, 3) as i8;
                                *q_ptr.add(i * 4) = v0;
                                *q_ptr.add(i * 4 + 1) = v1;
                                *q_ptr.add(i * 4 + 2) = v2;
                                *q_ptr.add(i * 4 + 3) = v3;
                            }
                        }
                    }

                    #[cfg(target_arch = "x86_64")]
                    {
                        // x86 AVX2 Implementation
                        let mut max_v = _mm256_setzero_ps();
                        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

                        for i in 0..4 {
                            let val = _mm256_loadu_ps(src.as_ptr().add(i * 8));
                            let abs_val = _mm256_and_ps(val, abs_mask);
                            max_v = _mm256_max_ps(max_v, abs_val);
                        }

                        let mut arr = [0.0f32; 8];
                        _mm256_storeu_ps(arr.as_mut_ptr(), max_v);
                        let mut max_abs = 0.0;
                        for &v in &arr {
                            if v > max_abs {
                                max_abs = v;
                            }
                        }

                        if max_abs == 0.0 {
                            *s_blk = 0.0;
                            std::ptr::write_bytes(q_blk.as_mut_ptr(), 0, 32);
                        } else {
                            let scale = max_abs / 127.0;
                            let inv_scale = 1.0 / scale;
                            *s_blk = scale;

                            for j in 0..32 {
                                let v = src[j] * inv_scale;
                                q_blk[j] = v.round().clamp(-127.0, 127.0) as i8;
                            }
                        }
                    }

                    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                    {
                        let mut max_abs = 0.0f32;
                        for &x in src {
                            let abs = x.abs();
                            if abs > max_abs {
                                max_abs = abs;
                            }
                        }
                        if max_abs == 0.0 {
                            *s_blk = 0.0;
                            q_blk.fill(0);
                        } else {
                            let scale = max_abs / 127.0;
                            let inv_scale = 1.0 / scale;
                            *s_blk = scale;
                            for i in 0..block_size {
                                let v = (src[i] * inv_scale).round();
                                q_blk[i] = v.clamp(-127.0, 127.0) as i8;
                            }
                        }
                    }
                }
            });
    }

    // -------------------------------------------------------------------------
    // 3. ARM NEON Kernel (SDOT) - [NIGHTLY / FIXED]
    // -------------------------------------------------------------------------
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[target_feature(enable = "dotprod")] // [중요] Nightly Feature
    unsafe fn kernel_neon_sdot(
        res_chunk: &mut [f32],
        start_col: usize,
        k: usize,
        num_blocks: usize,
        b_q: &[u8],
        b_s: &[f32],
        a_q8: &[i8],
        a_scales: &[f32],
    ) {
        let v_mask_low = vdupq_n_u8(0x0F);
        let v_minus_8 = vdupq_n_s8(-8);

        for (i, res) in res_chunk.iter_mut().enumerate() {
            let j = start_col + i;
            let mut q4_ptr = b_q.as_ptr().add(j * (k / 2));
            let mut s_b_ptr = b_s.as_ptr().add(j * num_blocks);
            let mut q8_ptr = a_q8.as_ptr();
            let mut s_a_ptr = a_scales.as_ptr();

            let mut total_sum = 0.0f32;

            for _ in 0..(num_blocks / 2) {
                let scale_1 = *s_b_ptr * *s_a_ptr;
                let scale_2 = *s_b_ptr.add(1) * *s_a_ptr.add(1);

                // Load Q4 (32 weights = 16 bytes)
                let q4_raw = vld1q_u8(q4_ptr);

                // 1. Unpack Low/High Nibbles
                let v_low_u8 = vandq_u8(q4_raw, v_mask_low);
                let v_high_u8 = vshrq_n_u8(q4_raw, 4);

                // 2. Adjust Weights (w - 8)
                let w_low_s8 = vaddq_s8(vreinterpretq_s8_u8(v_low_u8), v_minus_8);
                let w_high_s8 = vaddq_s8(vreinterpretq_s8_u8(v_high_u8), v_minus_8);

                // 3. [FIX] Interleave Weights to match Activation order
                // 기존 문제: Even/Odd 분리로 인한 순서 불일치 해결
                let w_0_15 = vzip1q_s8(w_low_s8, w_high_s8); // w0..w15
                let w_16_31 = vzip2q_s8(w_low_s8, w_high_s8); // w16..w31

                // Load Activations
                let a_0 = vld1q_s8(q8_ptr); // a0..a15
                let a_1 = vld1q_s8(q8_ptr.add(16)); // a16..a31

                // 4. SDOT (Hardware Accelerated)
                // vdotq_s32는 레지스터 내 4개 원소씩 곱합을 수행
                let mut acc = vdupq_n_s32(0);
                acc = vdotq_s32(acc, w_0_15, a_0);
                let sum1 = vaddvq_s32(acc) as f32;

                let mut acc2 = vdupq_n_s32(0);
                acc2 = vdotq_s32(acc2, w_16_31, a_1);
                let sum2 = vaddvq_s32(acc2) as f32;

                total_sum += sum1 * scale_1 + sum2 * scale_2;

                q4_ptr = q4_ptr.add(32);
                q8_ptr = q8_ptr.add(32);
                s_b_ptr = s_b_ptr.add(2);
                s_a_ptr = s_a_ptr.add(2);
            }
            *res = total_sum;
        }
    }

    // -------------------------------------------------------------------------
    // 4. x86 AVX2 Kernel - [FIXED]
    // -------------------------------------------------------------------------
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn kernel_avx2(
        res_chunk: &mut [f32],
        start_col: usize,
        k: usize,
        num_blocks: usize,
        b_q: &[u8],
        b_s: &[f32],
        a_q8: &[i8],
        a_scales: &[f32],
    ) {
        let ones = _mm256_set1_epi16(1);
        let ones_u8 = _mm256_set1_epi8(1);

        for (i, res) in res_chunk.iter_mut().enumerate() {
            let j = start_col + i;
            let mut q4_ptr = b_q.as_ptr().add(j * (k / 2));
            let mut s_b_ptr = b_s.as_ptr().add(j * num_blocks);
            let mut q8_ptr = a_q8.as_ptr();
            let mut s_a_ptr = a_scales.as_ptr();

            let mut total_sum = 0.0f32;

            for _ in 0..num_blocks {
                let scale = *s_b_ptr * *s_a_ptr;

                let q4_128 = _mm_loadu_si128(q4_ptr as *const _);

                // 1. Unpack Low/High Nibbles
                let low_128 = _mm_and_si128(q4_128, _mm_set1_epi8(0x0F));
                let high_128 = _mm_and_si128(_mm_srli_epi16(q4_128, 4), _mm_set1_epi8(0x0F));

                // 2. [FIX] Interleave using Unpack
                let w_lo = _mm_unpacklo_epi8(low_128, high_128);
                let w_hi = _mm_unpackhi_epi8(low_128, high_128);
                let w_u8_256 = _mm256_set_m128i(w_hi, w_lo);

                let a_256 = _mm256_loadu_si256(q8_ptr as *const _);

                // Term 1: sum(w * a)
                let dot_prod = _mm256_maddubs_epi16(w_u8_256, a_256);
                let sum_i32 = _mm256_madd_epi16(dot_prod, ones);

                // Term 2: sum(a) * 8
                let sum_a_vec = _mm256_maddubs_epi16(ones_u8, a_256);
                let sum_a_i32 = _mm256_madd_epi16(sum_a_vec, ones);

                // Final: Term1 - 8 * Term2
                let final_vec = _mm256_sub_epi32(sum_i32, _mm256_slli_epi32(sum_a_i32, 3));

                // Horizontal Reduction
                let v_perm = _mm256_permute2f128_si256(final_vec, final_vec, 1);
                let v_add = _mm256_add_epi32(final_vec, v_perm);
                let v_add = _mm256_hadd_epi32(v_add, v_add);
                let v_add = _mm256_hadd_epi32(v_add, v_add);
                let block_sum = _mm_cvtsi128_si32(_mm256_castsi256_si128(v_add)) as f32;

                total_sum += block_sum * scale;

                q4_ptr = q4_ptr.add(16);
                q8_ptr = q8_ptr.add(32);
                s_b_ptr = s_b_ptr.add(1);
                s_a_ptr = s_a_ptr.add(1);
            }
            *res = total_sum;
        }
    }
}
