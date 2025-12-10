use std::sync::Arc;

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
        "Rayon CPU"
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

        // Storage에서 데이터 슬라이스 추출 (CPU 백엔드이므로 CPU Storage여야 함)
        let a_data = match &*a.storage {
            Storage::Cpu(d) => d,
            _ => panic!("CpuBackend requires CPU storage for input A"),
        };
        let b_data = match &*b.storage {
            Storage::Cpu(d) => d,
            _ => panic!("CpuBackend requires CPU storage for input B"),
        };

        // Decoding 최적화 (M=1): 결과 벡터의 각 원소를 병렬 계산
        if m == 1 {
            result_data.par_iter_mut().enumerate().for_each(|(j, res)| {
                let mut sum = 0.0;
                for p in 0..k {
                    unsafe {
                        // B는 열(Column) 방향 접근이므로 캐시 효율은 낮지만, 로직은 정확함
                        sum += *a_data.get_unchecked(p) * *b_data.get_unchecked(p * n + j);
                    }
                }
                *res = sum;
            });
        } else {
            // Prefill 최적화 (M > 1): 결과 행렬의 행(Row) 단위로 병렬화
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

    // [전치 행렬 곱] A @ B^T (B가 이미 전치되어 저장된 경우)
    fn matmul_transposed(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let dims_a = a.shape().dims();
        let dims_b_t = b.shape().dims();

        let (m, k) = (dims_a[0], dims_a[1]);
        let (n, k2) = (dims_b_t[0], dims_b_t[1]);

        if k != k2 {
            panic!(
                "MatMul Transposed mismatch: A[{},{}] vs B_t[{},{}]",
                m, k, n, k2
            );
        }

        let mut result_data = vec![0.0; m * n];
        let a_data = match &*a.storage {
            Storage::Cpu(d) => d,
            _ => panic!("CpuBackend requires CPU storage"),
        };
        let b_data = match &*b.storage {
            Storage::Cpu(d) => d,
            _ => panic!("CpuBackend requires CPU storage"),
        };

        if m == 1 {
            // M=1 최적화: 결과 벡터 병렬화
            result_data.par_iter_mut().enumerate().for_each(|(j, res)| {
                let a_slice = &a_data[0..k];
                let b_row_start = j * k;
                let b_slice = &b_data[b_row_start..b_row_start + k];

                // 연속 메모리 내적 (SIMD 가속됨)
                let sum: f32 = a_slice.iter().zip(b_slice.iter()).map(|(x, y)| x * y).sum();
                *res = sum;
            });
        } else {
            // M > 1 최적화: 행 단위 병렬화
            result_data
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, res_row)| {
                    let a_row_start = i * k;
                    let a_slice = &a_data[a_row_start..a_row_start + k];
                    for j in 0..n {
                        let b_row_start = j * k;
                        let b_slice = &b_data[b_row_start..b_row_start + k];
                        let sum: f32 = a_slice.iter().zip(b_slice.iter()).map(|(x, y)| x * y).sum();
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

    // [In-Place 덧셈]
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

    // [Fused SiLU + Mul]
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

    // [RoPE In-Place]
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

    // [RMS Norm]
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

        let mut output_data = vec![0.0; x_data.len()];

        output_data
            .par_chunks_mut(last_dim)
            .zip(x_data.par_chunks(last_dim))
            .for_each(|(out_row, in_row)| {
                let ss: f32 = in_row.iter().map(|v| v * v).sum();
                let rms = (ss / last_dim as f32 + eps).sqrt();
                let scale = 1.0 / rms;

                for j in 0..last_dim {
                    out_row[j] = in_row[j] * scale * w_data[j];
                }
            });

        Tensor::new(output_data, x.shape().clone())
    }

    // [Softmax]
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

    // [복사/클론]
    fn copy_from(&self, tensor: &Tensor) -> Tensor {
        match &*tensor.storage {
            Storage::Cpu(data) => Tensor::new(data.clone(), tensor.shape().clone()),
            _ => panic!("Copying from non-CPU storage not implemented yet"),
        }
    }
}
