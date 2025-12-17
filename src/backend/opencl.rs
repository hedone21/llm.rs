use crate::backend::{Backend, Device};
use crate::core::shape::Shape;
use crate::core::tensor::{Storage, Tensor};
use ocl::{Buffer, ProQue, flags};
use std::sync::{Arc, OnceLock};

static CL_CONTEXT: OnceLock<Arc<ProQue>> = OnceLock::new();

pub struct OpenClBackend {
    pub context: Arc<ProQue>,
}

impl OpenClBackend {
    pub fn new() -> Self {
        let context = CL_CONTEXT.get_or_init(|| {
            // [Phase 3] Full Offloading을 위한 커널 세트 추가
            let src = r#"
                __kernel void add_dummy(__global float* a) {
                    int id = get_global_id(0);
                    a[id] += 1.0f;
                }

                __kernel void matmul(
                    __global const float* A,
                    __global const float* B,
                    __global float* C,
                    const int M,
                    const int K,
                    const int N)
                {
                    int row = get_global_id(0);
                    int col = get_global_id(1);

                    if (row < M && col < N) {
                        float sum = 0.0f;
                        for (int k = 0; k < K; ++k) {
                            sum += A[row * K + k] * B[k * N + col];
                        }
                        C[row * N + col] = sum;
                    }
                }

                __kernel void matmul_transposed(
                    __global const float* A,
                    __global const float* B,
                    __global float* C,
                    const int M,
                    const int K,
                    const int N)
                {
                    int row = get_global_id(0);
                    int col = get_global_id(1);

                    if (row < M && col < N) {
                        float sum = 0.0f;
                        for (int k = 0; k < K; ++k) {
                            sum += A[row * K + k] * B[col * K + k];
                        }
                        C[row * N + col] = sum;
                    }
                }

                // [New] In-place Add: A += B
                __kernel void add_assign(__global float* a, __global const float* b, const int n) {
                    int i = get_global_id(0);
                    if (i < n) {
                        a[i] += b[i];
                    }
                }

                // [New] SiLU * Mul: Gate = (Gate / (1 + exp(-Gate))) * Up
                __kernel void silu_mul(__global float* gate, __global const float* up, const int n) {
                    int i = get_global_id(0);
                    if (i < n) {
                        float g = gate[i];
                        float silu = g / (1.0f + exp(-g));
                        gate[i] = silu * up[i];
                    }
                }

                // [New] Scalar Scale: x *= val
                __kernel void scale(__global float* x, const float val, const int n) {
                    int i = get_global_id(0);
                    if (i < n) {
                        x[i] *= val;
                    }
                }

                // [New] RMS Norm
                // 간단한 구현: 1 Thread per Row (병렬성은 Row 개수만큼)
                __kernel void rms_norm(
                    __global const float* x,
                    __global const float* weight,
                    __global float* out,
                    const int row_stride, // Hidden Size
                    const float eps)
                {
                    int row = get_global_id(0);
                    int offset = row * row_stride;

                    float ss = 0.0f;
                    for (int i = 0; i < row_stride; i++) {
                        float v = x[offset + i];
                        ss += v * v;
                    }
                    
                    float rms = sqrt(ss / (float)row_stride + eps);
                    float scale = 1.0f / rms;

                    for (int i = 0; i < row_stride; i++) {
                        out[offset + i] = x[offset + i] * scale * weight[i];
                    }
                }

                // [New] Softmax
                // 간단한 구현: 1 Thread per Row
                __kernel void softmax(
                    __global const float* x,
                    __global float* out,
                    const int row_stride)
                {
                    int row = get_global_id(0);
                    int offset = row * row_stride;

                    // 1. Max (for stability)
                    float max_val = -1e30f; // -infinity
                    for (int i = 0; i < row_stride; i++) {
                        float v = x[offset + i];
                        if (v > max_val) max_val = v;
                    }

                    // 2. Exp Sum
                    float sum_exp = 0.0f;
                    for (int i = 0; i < row_stride; i++) {
                        float e = exp(x[offset + i] - max_val);
                        out[offset + i] = e; // 임시 저장
                        sum_exp += e;
                    }

                    // 3. Normalize
                    float inv_sum = 1.0f / sum_exp;
                    for (int i = 0; i < row_stride; i++) {
                        out[offset + i] *= inv_sum;
                    }
                }

                // [New] RoPE (Rotary Positional Embedding)
                // x: [Seq, HeadDim] (Flat)
                // Thread는 (Seq * HeadDim / 2) 만큼 실행
                __kernel void rope_inplace(
                    __global float* x,
                    const int start_pos,
                    const int head_dim,
                    const float theta_base)
                {
                    int gid = get_global_id(0);
                    
                    int mid = head_dim / 2;
                    int row = gid / mid;      // Sequence Index
                    int j = gid % mid;        // Pair Index (0 ~ mid-1)

                    int pos = start_pos + row;
                    
                    // Frequency Calculation
                    float freq_idx = ((float)j * 2.0f) / (float)head_dim;
                    float theta = 1.0f / pow(theta_base, freq_idx);
                    float m_theta = (float)pos * theta;
                    
                    float sin_val = sin(m_theta);
                    float cos_val = cos(m_theta);

                    int idx_re = row * head_dim + j;
                    int idx_im = idx_re + mid;

                    float re = x[idx_re];
                    float im = x[idx_im];

                    x[idx_re] = re * cos_val - im * sin_val;
                    x[idx_im] = re * sin_val + im * cos_val;
                }
            "#;

            let pro_que = ProQue::builder()
                .src(src)
                .dims(1)
                .build()
                .expect("Failed to initialize OpenCL");

            println!(
                "[OpenCL] Initialized on: {}",
                pro_que.device().name().unwrap_or("Unknown".into())
            );
            Arc::new(pro_que)
        });

        Self {
            context: context.clone(),
        }
    }

    pub fn allocate_shared(&self, shape: &Shape) -> Tensor {
        // (Phase 1, 2와 동일)
        let count = shape.num_elements();
        let size_bytes = count * 4;

        let buffer = Buffer::<f32>::builder()
            .queue(self.context.queue().clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_ALLOC_HOST_PTR)
            .len(count)
            .build()
            .expect("Failed to allocate shared buffer");

        let ptr = unsafe {
            buffer
                .map()
                .enq()
                .expect("Failed to map buffer")
                .as_mut_ptr()
        };

        Tensor::from_storage(
            Storage::Shared {
                ptr: ptr as *mut u8,
                size: size_bytes,
                cl_buffer: Some(buffer),
                queue: Some(self.context.queue().clone()),
                handle: 0,
            },
            shape.clone(),
            Device::OpenCl,
        )
    }

    pub fn launch_dummy_kernel(&self, tensor: &Tensor) {
        if let Storage::Shared {
            cl_buffer: Some(buffer),
            ..
        } = &*tensor.storage
        {
            let kernel = self
                .context
                .kernel_builder("add_dummy")
                .arg(buffer)
                .global_work_size(tensor.shape().num_elements())
                .build()
                .expect("Failed to build dummy kernel");
            unsafe {
                kernel.enq().expect("Failed to enqueue kernel");
            }
            self.context
                .queue()
                .finish()
                .expect("Failed to finish queue");
        }
    }

    fn get_buffer<'a>(&self, tensor: &'a Tensor) -> &'a Buffer<f32> {
        match &*tensor.storage {
            Storage::Shared {
                cl_buffer: Some(buf),
                ..
            } => buf,
            Storage::OpenCl(buf) => buf,
            _ => panic!("Tensor must be on OpenCL device (Shared or OpenCl)"),
        }
    }
}

impl Backend for OpenClBackend {
    fn device(&self) -> Device {
        Device::OpenCl
    }
    fn name(&self) -> &str {
        "ARM OpenCL (Zero-Copy)"
    }

    // [Phase 2]
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let dim_a = a.shape().dims();
        let dim_b = b.shape().dims();
        let (m, k) = (dim_a[0], dim_a[1]);
        let (k2, n) = (dim_b[0], dim_b[1]);
        assert_eq!(k, k2, "Matmul dimension mismatch");

        let c = self.allocate_shared(&Shape::new(vec![m, n]));
        let buf_a = self.get_buffer(a);
        let buf_b = self.get_buffer(b);
        let buf_c = self.get_buffer(&c);

        let kernel = self
            .context
            .kernel_builder("matmul")
            .arg(buf_a)
            .arg(buf_b)
            .arg(buf_c)
            .arg(m as i32)
            .arg(k as i32)
            .arg(n as i32)
            .global_work_size([m, n])
            .build()
            .expect("Failed to build matmul");
        unsafe {
            kernel.enq().expect("Failed to enq");
        }
        self.context.queue().finish().expect("Failed to finish");
        c
    }

    // [Phase 2]
    fn matmul_transposed(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let dim_a = a.shape().dims();
        let dim_b = b.shape().dims();
        let (m, k) = (dim_a[0], dim_a[1]);
        let (n, k2) = (dim_b[0], dim_b[1]);
        assert_eq!(k, k2, "Matmul T mismatch");

        let c = self.allocate_shared(&Shape::new(vec![m, n]));
        let buf_a = self.get_buffer(a);
        let buf_b = self.get_buffer(b);
        let buf_c = self.get_buffer(&c);

        let kernel = self
            .context
            .kernel_builder("matmul_transposed")
            .arg(buf_a)
            .arg(buf_b)
            .arg(buf_c)
            .arg(m as i32)
            .arg(k as i32)
            .arg(n as i32)
            .global_work_size([m, n])
            .build()
            .expect("Failed to build matmul_transposed");
        unsafe {
            kernel.enq().expect("Failed to enq");
        }
        self.context.queue().finish().expect("Failed to finish");
        c
    }

    // [Phase 3] CPU Fallback for Slice Ops
    fn matmul_slice(&self, a: &Tensor, other_data: &[f32], rows: usize, cols: usize) -> Tensor {
        // GPU 텐서 a는 Shared Memory이므로, CPU에서도 접근 가능합니다.
        // Slice는 버퍼 핸들이 없으므로 커널에 전달하기 어렵습니다.
        // 따라서 CPU Fallback을 사용합니다 (Shared Memory 덕분에 복사 비용 없음).
        let dims_a = a.shape().dims();
        let (m, k) = (dims_a[0], dims_a[1]);
        assert_eq!(k, cols, "MatMul Slice mismatch");

        let mut result_data = vec![0.0; m * rows];

        // Zero-Copy Access
        let a_data = a.data();

        // Simple CPU MatMul
        for i in 0..m {
            let a_row = &a_data[i * k..(i + 1) * k];
            for j in 0..rows {
                let b_row = &other_data[j * cols..(j + 1) * cols];
                let sum: f32 = a_row.iter().zip(b_row.iter()).map(|(x, y)| x * y).sum();
                result_data[i * rows + j] = sum;
            }
        }

        // 결과도 Shared Tensor로 반환
        let mut res_tensor = self.allocate_shared(&Shape::new(vec![m, rows]));
        unsafe {
            // Memory Copy (CPU -> Shared)
            std::ptr::copy_nonoverlapping(
                result_data.as_ptr(),
                res_tensor.data_mut().as_mut_ptr(),
                result_data.len(),
            );
        }
        res_tensor
    }

    // [Phase 3] In-place Add
    fn add_assign(&self, a: &mut Tensor, b: &Tensor) {
        let n = a.shape().num_elements();
        assert_eq!(n, b.shape().num_elements(), "Shape mismatch add_assign");

        let buf_a = self.get_buffer(a);
        let buf_b = self.get_buffer(b);

        let kernel = self
            .context
            .kernel_builder("add_assign")
            .arg(buf_a)
            .arg(buf_b)
            .arg(n as i32)
            .global_work_size(n)
            .build()
            .expect("Failed to build add_assign");

        unsafe {
            kernel.enq().expect("Failed to enq add_assign");
        }
        self.context.queue().finish().expect("Failed to finish");
    }

    // [Phase 3] SiLU * Mul
    fn silu_mul(&self, gate: &mut Tensor, up: &Tensor) {
        let n = gate.shape().num_elements();
        assert_eq!(n, up.shape().num_elements());

        let buf_g = self.get_buffer(gate);
        let buf_u = self.get_buffer(up);

        let kernel = self
            .context
            .kernel_builder("silu_mul")
            .arg(buf_g)
            .arg(buf_u)
            .arg(n as i32)
            .global_work_size(n)
            .build()
            .expect("Failed to build silu_mul");

        unsafe {
            kernel.enq().expect("Failed to enq silu_mul");
        }
        self.context.queue().finish().expect("Failed to finish");
    }

    // [Phase 3] RoPE
    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize) {
        let dims = x.shape().dims();
        // x shape assumption: [Seq, HeadDim] (flattened from [Heads, Seq, HeadDim] in previous steps)
        // Or simply num_elements / head_dim (since RoPE is applied to last dim)
        let head_dim = *dims.last().unwrap();
        let total_elements = x.shape().num_elements();
        let seq_len = total_elements / head_dim; // Total tokens involved (Batch * Heads * Seq)

        // Work items: Total Pairs to rotate
        let total_pairs = total_elements / 2;

        let buf_x = self.get_buffer(x);
        let theta_base = 500_000.0f32;

        let kernel = self
            .context
            .kernel_builder("rope_inplace")
            .arg(buf_x)
            .arg(start_pos as i32)
            .arg(head_dim as i32)
            .arg(theta_base)
            .global_work_size(total_pairs)
            .build()
            .expect("Failed to build rope_inplace");

        unsafe {
            kernel.enq().expect("Failed to enq rope");
        }
        self.context.queue().finish().expect("Failed to finish");
    }

    // [Phase 3] RMS Norm
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
        let dims = x.shape().dims();
        let row_stride = *dims.last().unwrap();
        let num_rows = x.shape().num_elements() / row_stride;

        let out = self.allocate_shared(x.shape());

        let buf_x = self.get_buffer(x);
        let buf_w = self.get_buffer(weight);
        let buf_out = self.get_buffer(&out);

        let kernel = self
            .context
            .kernel_builder("rms_norm")
            .arg(buf_x)
            .arg(buf_w)
            .arg(buf_out)
            .arg(row_stride as i32)
            .arg(eps)
            .global_work_size(num_rows) // 1 Thread per Row
            .build()
            .expect("Failed to build rms_norm");

        unsafe {
            kernel.enq().expect("Failed to enq rms_norm");
        }
        self.context.queue().finish().expect("Failed to finish");
        out
    }

    // [Phase 3] Softmax
    fn softmax(&self, x: &Tensor) -> Tensor {
        let dims = x.shape().dims();
        let row_stride = *dims.last().unwrap();
        let num_rows = x.shape().num_elements() / row_stride;

        let out = self.allocate_shared(x.shape());

        let buf_x = self.get_buffer(x);
        let buf_out = self.get_buffer(&out);

        let kernel = self
            .context
            .kernel_builder("softmax")
            .arg(buf_x)
            .arg(buf_out)
            .arg(row_stride as i32)
            .global_work_size(num_rows) // 1 Thread per Row
            .build()
            .expect("Failed to build softmax");

        unsafe {
            kernel.enq().expect("Failed to enq softmax");
        }
        self.context.queue().finish().expect("Failed to finish");
        out
    }

    // [Phase 3] Scale
    fn scale(&self, x: &Tensor, value: f32) -> Tensor {
        // 1. 결과 텐서 할당
        let mut out = self.allocate_shared(x.shape());
        let n = x.shape().num_elements();

        // 2. [데이터 복사] CPU -> Shared Memory (Zero-Copy)
        // 먼저 데이터를 복사합니다. 이때는 out의 Mutable borrow가 필요합니다.
        unsafe {
            let src = x.data();
            let dst = out.data_mut(); // Mutable borrow 발생
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), n);
        } // Mutable borrow 종료

        // 3. [버퍼 획득]
        // 데이터 복사가 끝났으므로 이제 안전하게 불변 참조(Buffer)를 얻을 수 있습니다.
        let buf_out = self.get_buffer(&out);

        // 4. [커널 실행] In-place Scaling
        let kernel = self
            .context
            .kernel_builder("scale")
            .arg(buf_out) // In-place on 'out'
            .arg(value)
            .arg(n as i32)
            .global_work_size(n)
            .build()
            .expect("Failed to build scale");

        unsafe {
            kernel.enq().expect("Failed to enq scale");
        }
        self.context.queue().finish().expect("Failed to finish");

        out
    }

    // [Phase 3] Copy From
    fn copy_from(&self, tensor: &Tensor) -> Tensor {
        // Deep copy
        let mut out = self.allocate_shared(tensor.shape());
        let n = tensor.shape().num_elements();
        unsafe {
            let src = tensor.data();
            let dst = out.data_mut();
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), n);
        }
        out
    }
}
