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
        "Cpu"
    }

    // -------------------------------------------------------------------------
    // 1. MatMul Transposed (A @ B^T)
    // 주 사용처: Linear Layer (Weights are transposed)
    // -------------------------------------------------------------------------
    fn matmul_transposed(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let dims_a = a.shape().dims();
        let dims_b_t = b.shape().dims();
        let (m, k) = (dims_a[0], dims_a[1]);
        let (n, k2) = (dims_b_t[0], dims_b_t[1]);

        assert_eq!(k, k2, "MatMul Transposed dimensions mismatch");

        let mut result_data = vec![0.0; m * n];

        let a_data = match &*a.storage {
            Storage::Cpu(d) => d.as_slice(),
            Storage::Shared { ptr, size, .. } => unsafe {
                std::slice::from_raw_parts(*ptr as *const f32, size / 4)
            },
            _ => panic!("Input A must be F32 (Cpu or Shared)"),
        };

        match &*b.storage {
            // [Case 1] Q4 Quantized Weights (기존 최적화 유지 + 버그 수정)
            Storage::CpuQ4 { .. } | Storage::SharedQ4 { .. } => {
                let (b_q_slice, b_s_slice) = match &*b.storage {
                    Storage::CpuQ4 { data, scales } => (data.as_slice(), scales.as_slice()),
                    Storage::SharedQ4 {
                        ptr,
                        data_len,
                        scale_len,
                        ..
                    } => unsafe {
                        (
                            std::slice::from_raw_parts(*ptr as *const u8, *data_len),
                            std::slice::from_raw_parts(
                                (*ptr).add(*data_len) as *const f32,
                                *scale_len,
                            ),
                        )
                    },
                    _ => unreachable!(),
                };
                let block_size = 32;
                let num_blocks = k / block_size;

                // Chunk size tuning: 512 works well for typical hidden sizes
                let chunk_size = 512;

                if m == 1 {
                    let mut a_q8 = vec![0i8; k];
                    let mut a_scales = vec![0.0f32; num_blocks];
                    self.quantize_row_q8_0(a_data, &mut a_q8, &mut a_scales);

                    result_data.par_chunks_mut(chunk_size).enumerate().for_each(
                        |(cid, res_chunk)| {
                            let start_col = cid * chunk_size;
                            self.call_simd_kernel_q4(
                                res_chunk, start_col, k, num_blocks, b_q_slice, b_s_slice, &a_q8,
                                &a_scales,
                            );
                        },
                    );
                } else {
                    result_data
                        .par_chunks_mut(n)
                        .enumerate()
                        .for_each(|(i, res_row)| {
                            let a_row_start = i * k;
                            let a_row = &a_data[a_row_start..a_row_start + k];

                            let mut a_q8 = vec![0i8; k];
                            let mut a_scales = vec![0.0f32; num_blocks];
                            self.quantize_row_q8_0(a_row, &mut a_q8, &mut a_scales);

                            for (cid, res_chunk) in res_row.chunks_mut(chunk_size).enumerate() {
                                let start_col = cid * chunk_size;
                                self.call_simd_kernel_q4(
                                    res_chunk, start_col, k, num_blocks, b_q_slice, b_s_slice,
                                    &a_q8, &a_scales,
                                );
                            }
                        });
                }
            }

            // [Case 2] F32 Weights (Optimized with 8x Unrolling)
            s if matches!(s, Storage::Cpu(_) | Storage::Shared { .. }) => {
                let b_data = match s {
                    Storage::Cpu(d) => d.as_slice(),
                    Storage::Shared { ptr, size, .. } => unsafe {
                        std::slice::from_raw_parts(*ptr as *const f32, size / 4)
                    },
                    _ => unreachable!(),
                };

                if m == 1 {
                    // Matrix-Vector Multiplication -> Dot Product
                    result_data.par_iter_mut().enumerate().for_each(|(j, res)| {
                        let a_row = &a_data[0..k];
                        let b_row_start = j * k;
                        let b_row = &b_data[b_row_start..b_row_start + k];
                        *res = Self::simd_dot_f32(a_row, b_row);
                    });
                } else {
                    // Matrix-Matrix Multiplication
                    result_data
                        .par_chunks_mut(n)
                        .enumerate()
                        .for_each(|(i, res_row)| {
                            let a_row_start = i * k;
                            let a_row = &a_data[a_row_start..a_row_start + k];
                            for (j, res) in res_row.iter_mut().enumerate() {
                                let b_row_start = j * k;
                                let b_row = &b_data[b_row_start..b_row_start + k];
                                *res = Self::simd_dot_f32(a_row, b_row);
                            }
                        });
                }
            }
            _ => panic!("Unsupported storage type for B in matmul_transposed"),
        }

        Tensor::new(result_data, Shape::new(vec![m, n]))
    }

    // -------------------------------------------------------------------------
    // 2. MatMul Standard (A @ B)
    // 주 사용처: Attention Scores (Q @ K^T가 아님, 일반 매트멀)
    // -------------------------------------------------------------------------
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let dims_a = a.shape().dims();
        let dims_b = b.shape().dims();
        let (m, k) = (dims_a[0], dims_a[1]);
        let (k2, n) = (dims_b[0], dims_b[1]);

        assert_eq!(k, k2, "MatMul Dimension mismatch");

        let mut result_data = vec![0.0; m * n];

        let a_data = match &*a.storage {
            Storage::Cpu(d) => d.as_slice(),
            Storage::Shared { ptr, size, .. } => unsafe {
                std::slice::from_raw_parts(*ptr as *const f32, size / 4)
            },
            _ => panic!("CpuBackend requires CPU/Shared storage for input A"),
        };
        let b_data = match &*b.storage {
            Storage::Cpu(d) => d.as_slice(),
            Storage::Shared { ptr, size, .. } => unsafe {
                std::slice::from_raw_parts(*ptr as *const f32, size / 4)
            },
            _ => panic!("CpuBackend requires CPU/Shared storage for input B"),
        };

        // SAXPY based implementation (Outer Product style)
        if m == 1 {
            let chunk_size = 512;
            result_data.par_chunks_mut(chunk_size).enumerate().for_each(
                |(chunk_idx, res_chunk)| {
                    let col_offset = chunk_idx * chunk_size;
                    let chunk_len = res_chunk.len();

                    for p in 0..k {
                        let val = unsafe { *a_data.get_unchecked(p) };
                        if val == 0.0 {
                            continue;
                        }

                        let b_row_start = p * n + col_offset;
                        let b_slice = &b_data[b_row_start..b_row_start + chunk_len];
                        Self::simd_saxpy(val, b_slice, res_chunk);
                    }
                },
            );
        } else {
            result_data
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, res_row)| {
                    let a_row_start = i * k;
                    let a_row = &a_data[a_row_start..a_row_start + k];

                    for p in 0..k {
                        let val = unsafe { *a_row.get_unchecked(p) };
                        if val == 0.0 {
                            continue;
                        }

                        let b_row_start = p * n;
                        let b_slice = &b_data[b_row_start..b_row_start + n];
                        Self::simd_saxpy(val, b_slice, res_row);
                    }
                });
        }

        Tensor::new(result_data, Shape::new(vec![m, n]))
    }

    // -------------------------------------------------------------------------
    // 3. MatMul Slice (A @ Slice^T)
    // 주 사용처: KV Cache Attention (Zero-Copy)
    // -------------------------------------------------------------------------
    fn matmul_slice(&self, a: &Tensor, other_data: &[f32], rows: usize, cols: usize) -> Tensor {
        let dims_a = a.shape().dims();
        let (m, k) = (dims_a[0], dims_a[1]);

        assert_eq!(k, cols, "MatMul Slice mismatch");

        let mut result_data = vec![0.0; m * rows];
        let a_data = match &*a.storage {
            Storage::Cpu(d) => d.as_slice(),
            Storage::Shared { ptr, size, .. } => unsafe {
                std::slice::from_raw_parts(*ptr as *const f32, size / 4)
            },
            _ => panic!("CpuBackend requires CPU storage"),
        };

        if m == 1 {
            result_data.par_iter_mut().enumerate().for_each(|(j, res)| {
                let a_slice = &a_data[0..k];
                let b_row_start = j * cols;
                let b_slice = &other_data[b_row_start..b_row_start + cols];
                *res = Self::simd_dot_f32(a_slice, b_slice);
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
                        res_row[j] = Self::simd_dot_f32(a_slice, b_slice);
                    }
                });
        }

        Tensor::new(result_data, Shape::new(vec![m, rows]))
    }

    // -------------------------------------------------------------------------
    // Element-wise Operations
    // -------------------------------------------------------------------------
    fn add_assign(&self, a: &mut Tensor, b: &Tensor) {
        if a.shape() != b.shape() {
            panic!("Shape mismatch");
        }
        let a_data = match Arc::get_mut(&mut a.storage) {
            Some(Storage::Cpu(d)) => d,
            Some(Storage::Shared { ptr, size, .. }) => unsafe {
                std::slice::from_raw_parts_mut(*ptr as *mut f32, *size / 4)
            },
            _ => panic!("Cannot mutate shared/non-cpu storage"),
        };
        let b_data = match &*b.storage {
            Storage::Cpu(d) => d.as_slice(),
            Storage::Shared { ptr, size, .. } => unsafe {
                std::slice::from_raw_parts(*ptr as *const f32, size / 4)
            },
            _ => panic!("CpuBackend requires CPU storage"),
        };
        a_data
            .par_iter_mut()
            .zip(b_data.par_iter())
            .for_each(|(x, y)| *x += *y);
    }

    fn silu_mul(&self, gate: &mut Tensor, up: &Tensor) {
        if gate.shape() != up.shape() {
            panic!("Shape mismatch");
        }
        let gate_data = match Arc::get_mut(&mut gate.storage) {
            Some(Storage::Cpu(d)) => d,
            Some(Storage::Shared { ptr, size, .. }) => unsafe {
                std::slice::from_raw_parts_mut(*ptr as *mut f32, *size / 4)
            },
            _ => panic!("Cannot mutate"),
        };
        let up_data = match &*up.storage {
            Storage::Cpu(d) => d.as_slice(),
            Storage::Shared { ptr, size, .. } => unsafe {
                std::slice::from_raw_parts(*ptr as *const f32, size / 4)
            },
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
            Some(Storage::Shared { ptr, size, .. }) => unsafe {
                std::slice::from_raw_parts_mut(*ptr as *mut f32, *size / 4)
            },
            _ => panic!("Cannot mutate"),
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
            Storage::Cpu(d) => d.as_slice(),
            Storage::Shared { ptr, size, .. } => unsafe {
                std::slice::from_raw_parts(*ptr as *const f32, size / 4)
            },
            _ => panic!("CpuBackend requires CPU storage"),
        };
        let w_data = match &*weight.storage {
            Storage::Cpu(d) => d.as_slice(),
            Storage::Shared { ptr, size, .. } => unsafe {
                std::slice::from_raw_parts(*ptr as *const f32, size / 4)
            },
            _ => panic!("CpuBackend requires CPU storage"),
        };
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
            Storage::Cpu(d) => d.as_slice(),
            Storage::Shared { ptr, size, .. } => unsafe {
                std::slice::from_raw_parts(*ptr as *const f32, size / 4)
            },
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
            Storage::Cpu(d) => d.as_slice(),
            Storage::Shared { ptr, size, .. } => unsafe {
                std::slice::from_raw_parts(*ptr as *const f32, size / 4)
            },
            _ => panic!("CpuBackend requires CPU storage"),
        };
        let new_data: Vec<f32> = x_data.par_iter().map(|&e| e * value).collect();
        Tensor::new(new_data, x.shape().clone())
    }

    fn copy_from(&self, tensor: &Tensor) -> Tensor {
        match &*tensor.storage {
            Storage::Cpu(data) => Tensor::new(data.clone(), tensor.shape().clone()),
            Storage::Shared { ptr, size, .. } => {
                let slice = unsafe { std::slice::from_raw_parts(*ptr as *const f32, size / 4) };
                Tensor::new(slice.to_vec(), tensor.shape().clone())
            }
            _ => panic!("Copying from non-CPU/Shared storage not implemented yet"),
        }
    }
}

impl CpuBackend {
    // -------------------------------------------------------------------------
    // High-Performance SIMD Kernels (Llama.cpp Style)
    // -------------------------------------------------------------------------

    // 1. F32 Dot Product: sum(a * b)
    // Reference: ggml_vec_dot_f32 in vec.cpp
    // Strategy: Unroll 8x (32 elements per iteration) using 8 accumulators.
    #[inline(always)]
    fn simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0;

        #[cfg(target_arch = "aarch64")]
        unsafe {
            let n = a.len();
            let mut ptr_a = a.as_ptr();
            let mut ptr_b = b.as_ptr();

            // 8 independent accumulators to break dependency chain
            let mut sum0 = vdupq_n_f32(0.0);
            let mut sum1 = vdupq_n_f32(0.0);
            let mut sum2 = vdupq_n_f32(0.0);
            let mut sum3 = vdupq_n_f32(0.0);
            let mut sum4 = vdupq_n_f32(0.0);
            let mut sum5 = vdupq_n_f32(0.0);
            let mut sum6 = vdupq_n_f32(0.0);
            let mut sum7 = vdupq_n_f32(0.0);

            let mut i = 0;
            // Unroll 32 elements (8 vectors) per iteration
            while i + 32 <= n {
                let va0 = vld1q_f32(ptr_a);
                let vb0 = vld1q_f32(ptr_b);
                sum0 = vfmaq_f32(sum0, va0, vb0);

                let va1 = vld1q_f32(ptr_a.add(4));
                let vb1 = vld1q_f32(ptr_b.add(4));
                sum1 = vfmaq_f32(sum1, va1, vb1);

                let va2 = vld1q_f32(ptr_a.add(8));
                let vb2 = vld1q_f32(ptr_b.add(8));
                sum2 = vfmaq_f32(sum2, va2, vb2);

                let va3 = vld1q_f32(ptr_a.add(12));
                let vb3 = vld1q_f32(ptr_b.add(12));
                sum3 = vfmaq_f32(sum3, va3, vb3);

                let va4 = vld1q_f32(ptr_a.add(16));
                let vb4 = vld1q_f32(ptr_b.add(16));
                sum4 = vfmaq_f32(sum4, va4, vb4);

                let va5 = vld1q_f32(ptr_a.add(20));
                let vb5 = vld1q_f32(ptr_b.add(20));
                sum5 = vfmaq_f32(sum5, va5, vb5);

                let va6 = vld1q_f32(ptr_a.add(24));
                let vb6 = vld1q_f32(ptr_b.add(24));
                sum6 = vfmaq_f32(sum6, va6, vb6);

                let va7 = vld1q_f32(ptr_a.add(28));
                let vb7 = vld1q_f32(ptr_b.add(28));
                sum7 = vfmaq_f32(sum7, va7, vb7);

                ptr_a = ptr_a.add(32);
                ptr_b = ptr_b.add(32);
                i += 32;
            }

            // Reduce 8 vectors to 1
            // sum0 = sum0 + sum1 + ... + sum7
            let mut v_sum = vaddq_f32(
                vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)),
                vaddq_f32(vaddq_f32(sum4, sum5), vaddq_f32(sum6, sum7)),
            );

            // Leftovers: Process remaining in smaller chunks
            while i + 4 <= n {
                let va = vld1q_f32(ptr_a);
                let vb = vld1q_f32(ptr_b);
                v_sum = vfmaq_f32(v_sum, va, vb);
                ptr_a = ptr_a.add(4);
                ptr_b = ptr_b.add(4);
                i += 4;
            }

            // Horizontal Reduction
            sum = vaddvq_f32(v_sum);

            // Final scalar leftovers
            while i < n {
                sum += *ptr_a * *ptr_b;
                ptr_a = ptr_a.add(1);
                ptr_b = ptr_b.add(1);
                i += 1;
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            sum = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        }

        sum
    }

    // 2. SAXPY: Y += alpha * X
    // Strategy: Unroll 8x
    #[inline(always)]
    fn simd_saxpy(alpha: f32, x: &[f32], y: &mut [f32]) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let n = x.len();
            let mut ptr_x = x.as_ptr();
            let mut ptr_y = y.as_mut_ptr();
            let v_alpha = vdupq_n_f32(alpha);

            let mut i = 0;
            // Process 32 elements per loop (8 vectors)
            while i + 32 <= n {
                let vx0 = vld1q_f32(ptr_x);
                let vy0 = vld1q_f32(ptr_y);
                let res0 = vfmaq_f32(vy0, v_alpha, vx0);
                vst1q_f32(ptr_y, res0);

                let vx1 = vld1q_f32(ptr_x.add(4));
                let vy1 = vld1q_f32(ptr_y.add(4));
                let res1 = vfmaq_f32(vy1, v_alpha, vx1);
                vst1q_f32(ptr_y.add(4), res1);

                let vx2 = vld1q_f32(ptr_x.add(8));
                let vy2 = vld1q_f32(ptr_y.add(8));
                let res2 = vfmaq_f32(vy2, v_alpha, vx2);
                vst1q_f32(ptr_y.add(8), res2);

                let vx3 = vld1q_f32(ptr_x.add(12));
                let vy3 = vld1q_f32(ptr_y.add(12));
                let res3 = vfmaq_f32(vy3, v_alpha, vx3);
                vst1q_f32(ptr_y.add(12), res3);

                let vx4 = vld1q_f32(ptr_x.add(16));
                let vy4 = vld1q_f32(ptr_y.add(16));
                let res4 = vfmaq_f32(vy4, v_alpha, vx4);
                vst1q_f32(ptr_y.add(16), res4);

                let vx5 = vld1q_f32(ptr_x.add(20));
                let vy5 = vld1q_f32(ptr_y.add(20));
                let res5 = vfmaq_f32(vy5, v_alpha, vx5);
                vst1q_f32(ptr_y.add(20), res5);

                let vx6 = vld1q_f32(ptr_x.add(24));
                let vy6 = vld1q_f32(ptr_y.add(24));
                let res6 = vfmaq_f32(vy6, v_alpha, vx6);
                vst1q_f32(ptr_y.add(24), res6);

                let vx7 = vld1q_f32(ptr_x.add(28));
                let vy7 = vld1q_f32(ptr_y.add(28));
                let res7 = vfmaq_f32(vy7, v_alpha, vx7);
                vst1q_f32(ptr_y.add(28), res7);

                ptr_x = ptr_x.add(32);
                ptr_y = ptr_y.add(32);
                i += 32;
            }
            // Leftovers
            while i < n {
                *ptr_y += alpha * *ptr_x;
                ptr_x = ptr_x.add(1);
                ptr_y = ptr_y.add(1);
                i += 1;
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for (y_val, x_val) in y.iter_mut().zip(x.iter()) {
                *y_val += alpha * *x_val;
            }
        }
    }

    #[inline(always)]
    fn call_simd_kernel_q4(
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

    fn quantize_row_q8_0(&self, data: &[f32], out_q: &mut [i8], out_s: &mut [f32]) {
        let block_size = 32;
        out_q
            .par_chunks_mut(block_size)
            .zip(out_s.par_iter_mut())
            .zip(data.par_chunks(block_size))
            .for_each(|((q_blk, s_blk), src)| unsafe {
                #[cfg(target_arch = "aarch64")]
                {
                    let mut max_v = vdupq_n_f32(0.0);
                    let mut i = 0;
                    while i < block_size {
                        let val = vld1q_f32(src.as_ptr().add(i));
                        max_v = vmaxnmq_f32(max_v, vabsq_f32(val));
                        i += 4;
                    }
                    let max_abs = vmaxnmvq_f32(max_v);

                    if max_abs == 0.0 {
                        *s_blk = 0.0;
                        std::ptr::write_bytes(q_blk.as_mut_ptr(), 0, block_size);
                    } else {
                        let scale = max_abs / 127.0;
                        let inv_scale = 1.0 / scale;
                        *s_blk = scale;

                        let v_inv = vdupq_n_f32(inv_scale);
                        let q_ptr = q_blk.as_mut_ptr();

                        let mut k = 0;
                        while k < block_size {
                            let val0 = vld1q_f32(src.as_ptr().add(k));
                            let val1 = vld1q_f32(src.as_ptr().add(k + 4));
                            let val2 = vld1q_f32(src.as_ptr().add(k + 8));
                            let val3 = vld1q_f32(src.as_ptr().add(k + 12));

                            let s0 = vmulq_f32(val0, v_inv);
                            let s1 = vmulq_f32(val1, v_inv);
                            let s2 = vmulq_f32(val2, v_inv);
                            let s3 = vmulq_f32(val3, v_inv);

                            let r0 = vcvtaq_s32_f32(s0);
                            let r1 = vcvtaq_s32_f32(s1);
                            let r2 = vcvtaq_s32_f32(s2);
                            let r3 = vcvtaq_s32_f32(s3);

                            let n0 = vmovn_s32(r0);
                            let n1 = vmovn_s32(r1);
                            let n2 = vmovn_s32(r2);
                            let n3 = vmovn_s32(r3);

                            let p0 = vqmovn_s16(vcombine_s16(n0, n1));
                            let p1 = vqmovn_s16(vcombine_s16(n2, n3));

                            vst1_s8(q_ptr.add(k), p0);
                            vst1_s8(q_ptr.add(k + 8), p1);

                            k += 16;
                        }
                    }
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    let mut max_abs = 0.0f32;
                    for &x in src {
                        if x.abs() > max_abs {
                            max_abs = x.abs();
                        }
                    }
                    if max_abs == 0.0 {
                        *s_blk = 0.0;
                        q_blk.fill(0);
                    } else {
                        let scale = max_abs / 127.0;
                        *s_blk = scale;
                        let inv = 1.0 / scale;
                        for i in 0..block_size {
                            q_blk[i] = (src[i] * inv).round().clamp(-127.0, 127.0) as i8;
                        }
                    }
                }
            });
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[target_feature(enable = "dotprod")]
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

            // FIX: Use scalar accumulator to prevent 4x broadcasting error
            let mut total_scalar = 0.0f32;
            let mut b_idx = 0;

            // Unroll loop: process 4 blocks (128 weights) at a time
            while b_idx + 4 <= num_blocks {
                let va_s = vld1q_f32(s_a_ptr);
                let vb_s = vld1q_f32(s_b_ptr);
                let v_scales = vmulq_f32(va_s, vb_s);

                // Block 0
                let q4_0 = vld1q_u8(q4_ptr);
                let low0 = vandq_u8(q4_0, v_mask_low);
                let high0 = vshrq_n_u8(q4_0, 4);
                let w0_l = vaddq_s8(vreinterpretq_s8_u8(low0), v_minus_8);
                let w0_h = vaddq_s8(vreinterpretq_s8_u8(high0), v_minus_8);
                let w0 = vzip1q_s8(w0_l, w0_h);
                let w1 = vzip2q_s8(w0_l, w0_h);
                let a0 = vld1q_s8(q8_ptr);
                let a1 = vld1q_s8(q8_ptr.add(16));
                let mut acc0 = vdupq_n_s32(0);
                acc0 = vdotq_s32(acc0, w0, a0);
                acc0 = vdotq_s32(acc0, w1, a1);

                // Block 1
                let q4_1 = vld1q_u8(q4_ptr.add(16));
                let low1 = vandq_u8(q4_1, v_mask_low);
                let high1 = vshrq_n_u8(q4_1, 4);
                let w1_l = vaddq_s8(vreinterpretq_s8_u8(low1), v_minus_8);
                let w1_h = vaddq_s8(vreinterpretq_s8_u8(high1), v_minus_8);
                let w2 = vzip1q_s8(w1_l, w1_h);
                let w3 = vzip2q_s8(w1_l, w1_h);
                let a2 = vld1q_s8(q8_ptr.add(32));
                let a3 = vld1q_s8(q8_ptr.add(48));
                let mut acc1 = vdupq_n_s32(0);
                acc1 = vdotq_s32(acc1, w2, a2);
                acc1 = vdotq_s32(acc1, w3, a3);

                // Block 2
                let q4_2 = vld1q_u8(q4_ptr.add(32));
                let low2 = vandq_u8(q4_2, v_mask_low);
                let high2 = vshrq_n_u8(q4_2, 4);
                let w2_l = vaddq_s8(vreinterpretq_s8_u8(low2), v_minus_8);
                let w2_h = vaddq_s8(vreinterpretq_s8_u8(high2), v_minus_8);
                let w4 = vzip1q_s8(w2_l, w2_h);
                let w5 = vzip2q_s8(w2_l, w2_h);
                let a4 = vld1q_s8(q8_ptr.add(64));
                let a5 = vld1q_s8(q8_ptr.add(80));
                let mut acc2 = vdupq_n_s32(0);
                acc2 = vdotq_s32(acc2, w4, a4);
                acc2 = vdotq_s32(acc2, w5, a5);

                // Block 3
                let q4_3 = vld1q_u8(q4_ptr.add(48));
                let low3 = vandq_u8(q4_3, v_mask_low);
                let high3 = vshrq_n_u8(q4_3, 4);
                let w3_l = vaddq_s8(vreinterpretq_s8_u8(low3), v_minus_8);
                let w3_h = vaddq_s8(vreinterpretq_s8_u8(high3), v_minus_8);
                let w6 = vzip1q_s8(w3_l, w3_h);
                let w7 = vzip2q_s8(w3_l, w3_h);
                let a6 = vld1q_s8(q8_ptr.add(96));
                let a7 = vld1q_s8(q8_ptr.add(112));
                let mut acc3 = vdupq_n_s32(0);
                acc3 = vdotq_s32(acc3, w6, a6);
                acc3 = vdotq_s32(acc3, w7, a7);

                // Accumulate partial results to scalar
                let sum_i0 = vaddvq_s32(acc0) as f32;
                let sum_i1 = vaddvq_s32(acc1) as f32;
                let sum_i2 = vaddvq_s32(acc2) as f32;
                let sum_i3 = vaddvq_s32(acc3) as f32;

                let s0 = vgetq_lane_f32(v_scales, 0);
                let s1 = vgetq_lane_f32(v_scales, 1);
                let s2 = vgetq_lane_f32(v_scales, 2);
                let s3 = vgetq_lane_f32(v_scales, 3);

                // FIX: Add directly to scalar accumulator
                total_scalar += (sum_i0 * s0) + (sum_i1 * s1) + (sum_i2 * s2) + (sum_i3 * s3);

                q4_ptr = q4_ptr.add(64);
                q8_ptr = q8_ptr.add(128);
                s_b_ptr = s_b_ptr.add(4);
                s_a_ptr = s_a_ptr.add(4);
                b_idx += 4;
            }

            // Remainder loop
            while b_idx < num_blocks {
                let scale = *s_b_ptr * *s_a_ptr;
                let q4_raw = vld1q_u8(q4_ptr);
                let low = vandq_u8(q4_raw, v_mask_low);
                let high = vshrq_n_u8(q4_raw, 4);
                let w_l = vaddq_s8(vreinterpretq_s8_u8(low), v_minus_8);
                let w_h = vaddq_s8(vreinterpretq_s8_u8(high), v_minus_8);
                let w0 = vzip1q_s8(w_l, w_h);
                let w1 = vzip2q_s8(w_l, w_h);

                let a0 = vld1q_s8(q8_ptr);
                let a1 = vld1q_s8(q8_ptr.add(16));

                let mut acc = vdupq_n_s32(0);
                acc = vdotq_s32(acc, w0, a0);
                acc = vdotq_s32(acc, w1, a1);

                let sum_blk = vaddvq_s32(acc) as f32;
                total_scalar += sum_blk * scale;

                q4_ptr = q4_ptr.add(16);
                q8_ptr = q8_ptr.add(32);
                s_b_ptr = s_b_ptr.add(1);
                s_a_ptr = s_a_ptr.add(1);
                b_idx += 1;
            }
            *res = total_scalar;
        }
    }

    #[cfg(target_arch = "x86_64")]
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
        unsafe {
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
                    let low_128 = _mm_and_si128(q4_128, _mm_set1_epi8(0x0F));
                    let high_128 = _mm_and_si128(_mm_srli_epi16(q4_128, 4), _mm_set1_epi8(0x0F));
                    let w_lo = _mm_unpacklo_epi8(low_128, high_128);
                    let w_hi = _mm_unpackhi_epi8(low_128, high_128);
                    let w_u8_256 = _mm256_set_m128i(w_hi, w_lo);
                    let a_256 = _mm256_loadu_si256(q8_ptr as *const _);
                    let dot_prod = _mm256_maddubs_epi16(w_u8_256, a_256);
                    let sum_i32 = _mm256_madd_epi16(dot_prod, ones);
                    let sum_a_vec = _mm256_maddubs_epi16(ones_u8, a_256);
                    let sum_a_i32 = _mm256_madd_epi16(sum_a_vec, ones);
                    let final_vec = _mm256_sub_epi32(sum_i32, _mm256_slli_epi32(sum_a_i32, 3));
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
}
