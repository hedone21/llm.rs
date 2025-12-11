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

    // [수정된 MatMul Transposed] Hybrid 연산 지원
    fn matmul_transposed(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let dims_a = a.shape().dims();
        let dims_b_t = b.shape().dims();
        let (m, k) = (dims_a[0], dims_a[1]);
        let (n, k2) = (dims_b_t[0], dims_b_t[1]); // n=OutputDim, k2=InputDim

        if k != k2 {
            panic!("Dimension mismatch in matmul_transposed");
        }

        let mut result_data = vec![0.0; m * n];

        // 입력 A는 항상 F32라고 가정 (Activations)
        let a_data = match &*a.storage {
            Storage::Cpu(d) => d,
            _ => panic!("Input A must be F32 CPU Tensor"),
        };

        // 가중치 B의 타입에 따라 분기
        match &*b.storage {
            // Case 1: 가중치가 F32 (기존 로직)
            Storage::Cpu(b_data) => {
                // ... (기존 F32 구현 코드 그대로 유지) ...
                // 내용이 길어 생략합니다. 기존에 작성한 rayon 병렬 코드 그대로 두시면 됩니다.
                if m == 1 {
                    result_data.par_iter_mut().enumerate().for_each(|(j, res)| {
                        let a_slice = &a_data[0..k];
                        let b_slice = &b_data[j * k..(j + 1) * k];
                        *res = a_slice.iter().zip(b_slice).map(|(x, y)| x * y).sum();
                    });
                } else {
                    result_data
                        .par_chunks_mut(n)
                        .enumerate()
                        .for_each(|(i, res_row)| {
                            let a_slice = &a_data[i * k..(i + 1) * k];
                            for j in 0..n {
                                let b_slice = &b_data[j * k..(j + 1) * k];
                                res_row[j] = a_slice.iter().zip(b_slice).map(|(x, y)| x * y).sum();
                            }
                        });
                }
            }

            // Case 2: 가중치가 Q8 (양자화됨) -> Hybrid Kernel 실행
            Storage::CpuQ4 {
                data: b_q,
                scales: b_s,
            } => {
                let block_size = 32;
                let packed_block_size = 16;

                // 내부 연산용 매크로 (코드 중복 제거 및 인라인 강제)
                // 안전 검사를 끈(unsafe) 상태로 4바이트(8개 가중치)씩 처리
                macro_rules! compute_block_unsafe {
                    ($num_blocks:expr, $b_s:expr, $b_s_start:expr, $b_q:expr, $b_q_start:expr, $a_slice:expr, $a_start:expr) => {{
                        let mut total_sum = 0.0f32;

                        // Pointers for raw access
                        let s_ptr = $b_s.as_ptr().add($b_s_start);
                        let q_ptr = $b_q.as_ptr().add($b_q_start);
                        let a_ptr = $a_slice.as_ptr().add($a_start);

                        for bi in 0..$num_blocks {
                            let scale = *s_ptr.add(bi);

                            // 각 블록의 포인터 위치 계산
                            let curr_q_ptr = q_ptr.add(bi * packed_block_size);
                            let curr_a_ptr = a_ptr.add(bi * block_size);

                            let mut unscaled_sum = 0.0f32;

                            // [Loop Unrolling] 16번 반복 대신 4번 x 4바이트 처리
                            // 컴파일러가 SIMD 명령어로 변환하기 훨씬 유리함
                            for ii in 0..4 {
                                let base = ii * 4; // 0, 4, 8, 12

                                // 4 Bytes loading (8 weights)
                                let p0 = *curr_q_ptr.add(base);
                                let p1 = *curr_q_ptr.add(base + 1);
                                let p2 = *curr_q_ptr.add(base + 2);
                                let p3 = *curr_q_ptr.add(base + 3);

                                // Activations loading
                                let a_base = base * 2;
                                let a0 = *curr_a_ptr.add(a_base);
                                let a1 = *curr_a_ptr.add(a_base + 1);
                                let a2 = *curr_a_ptr.add(a_base + 2);
                                let a3 = *curr_a_ptr.add(a_base + 3);
                                let a4 = *curr_a_ptr.add(a_base + 4);
                                let a5 = *curr_a_ptr.add(a_base + 5);
                                let a6 = *curr_a_ptr.add(a_base + 6);
                                let a7 = *curr_a_ptr.add(a_base + 7);

                                // Computation
                                unscaled_sum += a0 * ((p0 & 0x0F) as f32 - 8.0);
                                unscaled_sum += a1 * ((p0 >> 4) as f32 - 8.0);
                                unscaled_sum += a2 * ((p1 & 0x0F) as f32 - 8.0);
                                unscaled_sum += a3 * ((p1 >> 4) as f32 - 8.0);
                                unscaled_sum += a4 * ((p2 & 0x0F) as f32 - 8.0);
                                unscaled_sum += a5 * ((p2 >> 4) as f32 - 8.0);
                                unscaled_sum += a6 * ((p3 & 0x0F) as f32 - 8.0);
                                unscaled_sum += a7 * ((p3 >> 4) as f32 - 8.0);
                            }
                            total_sum += unscaled_sum * scale;
                        }
                        total_sum
                    }};
                }

                // Remainder 처리는 로직이 복잡하므로 안전한 기존 방식 유지 (성능 영향 미미)
                // 대신 Pointer만 사용하여 조금 더 최적화
                let compute_remainder =
                    |remainder: usize, scale: f32, q_slice: &[u8], a_slice: &[f32]| -> f32 {
                        let mut unscaled_sum = 0.0f32;
                        let remainder_bytes = (remainder + 1) / 2;
                        for ii in 0..remainder_bytes {
                            unsafe {
                                let packed = *q_slice.get_unchecked(ii);
                                let q0 = (packed & 0x0F) as f32 - 8.0;
                                let q1 = (packed >> 4) as f32 - 8.0;
                                if ii * 2 < remainder {
                                    unscaled_sum += *a_slice.get_unchecked(ii * 2) * q0;
                                }
                                if ii * 2 + 1 < remainder {
                                    unscaled_sum += *a_slice.get_unchecked(ii * 2 + 1) * q1;
                                }
                            }
                        }
                        unscaled_sum * scale
                    };

                // --- Execution Logic ---

                if m == 1 {
                    result_data.par_iter_mut().enumerate().for_each(|(j, res)| {
                        let num_blocks = k / block_size;
                        let remainder = k % block_size;
                        let b_row_byte_start = j * (k / 2);
                        let b_scale_start = j * ((k + block_size - 1) / block_size);

                        unsafe {
                            // 1. Main Loop (Unsafe Optimized)
                            let mut sum = compute_block_unsafe!(
                                num_blocks,
                                b_s,
                                b_scale_start,
                                b_q,
                                b_row_byte_start,
                                a_data,
                                0
                            );

                            // 2. Remainder
                            if remainder > 0 {
                                let scale = *b_s.get_unchecked(b_scale_start + num_blocks);
                                let q_start = b_row_byte_start + num_blocks * packed_block_size;
                                let a_start = num_blocks * block_size;

                                // Slice creation for remainder function
                                let q_chunk = std::slice::from_raw_parts(
                                    b_q.as_ptr().add(q_start),
                                    (remainder + 1) / 2,
                                );
                                let a_chunk = std::slice::from_raw_parts(
                                    a_data.as_ptr().add(a_start),
                                    remainder,
                                );

                                sum += compute_remainder(remainder, scale, q_chunk, a_chunk);
                            }
                            *res = sum;
                        }
                    });
                } else {
                    // M > 1 (Prefill)
                    result_data
                        .par_chunks_mut(n)
                        .enumerate()
                        .for_each(|(i, res_row)| {
                            let a_row_offset = i * k; // Activation Row Offset

                            for j in 0..n {
                                let num_blocks = k / block_size;
                                let remainder = k % block_size;
                                let b_row_byte_start = j * (k / 2);
                                let b_scale_start = j * ((k + block_size - 1) / block_size);

                                unsafe {
                                    let mut sum = compute_block_unsafe!(
                                        num_blocks,
                                        b_s,
                                        b_scale_start,
                                        b_q,
                                        b_row_byte_start,
                                        a_data,
                                        a_row_offset
                                    );

                                    if remainder > 0 {
                                        let scale = *b_s.get_unchecked(b_scale_start + num_blocks);
                                        let q_start =
                                            b_row_byte_start + num_blocks * packed_block_size;
                                        let a_start = a_row_offset + num_blocks * block_size;

                                        let q_chunk = std::slice::from_raw_parts(
                                            b_q.as_ptr().add(q_start),
                                            (remainder + 1) / 2,
                                        );
                                        let a_chunk = std::slice::from_raw_parts(
                                            a_data.as_ptr().add(a_start),
                                            remainder,
                                        );

                                        sum +=
                                            compute_remainder(remainder, scale, q_chunk, a_chunk);
                                    }
                                    res_row[j] = sum;
                                }
                            }
                        });
                }
            }

            _ => panic!("Unsupported storage types for matmul_transposed"),
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

        // 미리 검증하여 루프 내부의 불확실성 제거
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
                // 1. Sum of Squares (Auto-Vectorization Friendly)
                // iter().fold()는 컴파일러가 SIMD 환원(Reduction)을 적용하기 가장 좋은 패턴입니다.
                let ss: f32 = in_row.iter().fold(0.0, |acc, &v| acc + v * v);

                // 2. RMS Calculation
                // rsqrt(역수 제곱근) 근사 명령어를 쓰지 않고 정석대로 계산하되,
                // ss 계산과 분리하여 파이프라인 효율을 높입니다.
                let rms = (ss / last_dim as f32 + eps).sqrt();
                let scale = 1.0 / rms;

                // 3. Normalization + Weight Application (Bounds Check Removed)
                // 인덱싱(out_row[j]) 대신 zip을 사용하여 경계 검사를 원천 차단합니다.
                // w_data는 모든 행(Row)에서 재사용되므로 L1 캐시에 남아있어 매우 빠릅니다.
                out_row
                    .iter_mut()
                    .zip(in_row.iter())
                    .zip(w_data.iter())
                    .for_each(|((out, &x_val), &w_val)| {
                        // FMA (Fused Multiply-Add) 최적화 가능성 높음
                        *out = x_val * scale * w_val;
                    });
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
