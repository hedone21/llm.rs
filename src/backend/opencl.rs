use crate::backend::{Backend, Device};
use crate::core::shape::Shape;
use crate::core::tensor::{Storage, Tensor};
use log::*;
use ocl::{Buffer, ProQue, flags};
use std::sync::{Mutex, OnceLock};

static CL_CONTEXT: OnceLock<ProQue> = OnceLock::new();
static SCRATCH_POOL: OnceLock<Mutex<ScratchPool>> = OnceLock::new();

pub fn cl_context() -> &'static ProQue {
    CL_CONTEXT.get_or_init(|| {
        // [Step 1] Load Kernels from files
        // Assumes 'kernels' directory is at the project root
        let src = format!(
            "{}\n{}\n{}\n{}\n{}\n{}",
            include_str!("../../kernels/common.cl"),
            include_str!("../../kernels/rms_norm.cl"),
            include_str!("../../kernels/matmul_q4.cl"),
            include_str!("../../kernels/matmul_f32.cl"),
            include_str!("../../kernels/rope.cl"),
            include_str!("../../kernels/elementwise.cl"),
        );

        println!("[OpenCL] Loading Kernels from files...");

        ProQue::builder()
            .src(src)
            .dims(1) // Default dims
            .build()
            .expect("Failed to build OpenCL program. Check kernel syntax.")
    })
}

// Scratch Pool Structure
struct ScratchPool {
    buffer: Buffer<f32>, // 거대한 단일 버퍼
    ptr: *mut f32,       // 매핑된 호스트 포인터
    capacity: usize,     // 총 엘리먼트 개수 (f32 기준)
    offset: usize,       // 현재 사용 중인 오프셋
}

unsafe impl Send for ScratchPool {}

// 스크래치 풀 초기화 함수 (Pre-init 단계에서 호출)
pub fn init_scratch_pool(size_mb: usize) {
    let context = cl_context();
    // f32는 4바이트이므로 mb * 1024*1024 / 4
    let num_elements = (size_mb * 1024 * 1024) / 4;

    println!(
        "[ScratchPool] Allocating shared memory: {} MB ({} elements)",
        size_mb, num_elements
    );

    // 1. 거대한 버퍼 할당 (Zero-Copy)
    let buffer = Buffer::<f32>::builder()
        .queue(context.queue().clone())
        .flags(flags::MEM_READ_WRITE | flags::MEM_ALLOC_HOST_PTR)
        .len(num_elements)
        .build()
        .expect("Failed to allocate scratch pool");

    // 2. 미리 매핑(Mapping)하여 포인터 획득
    // 주의: Unmap을 하지 않고 유지합니다 (Persistent Mapping)
    let ptr = unsafe {
        buffer
            .map()
            .flags(flags::MAP_WRITE | flags::MAP_READ)
            .enq()
            .expect("Failed to map scratch buffer")
            .as_mut_ptr()
    };

    SCRATCH_POOL
        .set(Mutex::new(ScratchPool {
            buffer,
            ptr,
            capacity: num_elements,
            offset: 0,
        }))
        .ok();
}

// 스크래치 풀 리셋 (매 Step 종료 시 호출)
pub fn reset_scratch_pool() {
    if let Some(pool_lock) = SCRATCH_POOL.get() {
        let mut pool = pool_lock.lock().unwrap();
        pool.offset = 0;
    }
}

pub struct OpenClBackend {
    pub context: &'static ProQue,
    // [Performance] Optional: Scratch buffers for ping-pong buffering
    // pub scratch_a: Arc<Buffer<f32>>,
    // pub scratch_b: Arc<Buffer<f32>>,
}

impl Default for OpenClBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenClBackend {
    pub fn new() -> Self {
        Self {
            context: cl_context(),
        }
    }

    // --- Helper to get raw buffer ---
    fn get_buffer<'a>(&self, t: &'a Tensor) -> &'a Buffer<f32> {
        match &*t.storage {
            Storage::Shared {
                cl_buffer: Some(b), ..
            } => b,
            _ => panic!("Tensor is not F32 OpenCL backed"),
        }
    }

    // --- Allocators (Existing behavior maintained) ---
    pub fn allocate_shared(&self, shape: &Shape) -> Tensor {
        let count = shape.num_elements();

        if let Some(pool_lock) = SCRATCH_POOL.get() {
            let mut pool = pool_lock.lock().unwrap();

            // 정렬 기준 계산 (예: 128바이트 정렬 = f32 32개 단위)
            // 실제로는 장치 정보를 쿼리하여 CL_DEVICE_MEM_BASE_ADDR_ALIGN을 쓰는 것이 정확합니다.
            let alignment_elements = 32; // 128 bytes / 4 bytes per f32
            let aligned_offset =
                (pool.offset + alignment_elements - 1) / alignment_elements * alignment_elements;

            // 용량 체크
            if aligned_offset + count <= pool.capacity {
                pool.offset = aligned_offset + count; // 다음을 위해 업데이트
                let current_offset = aligned_offset;

                // 1. 호스트 포인터 계산 (단순 산술 연산, 비용 0)
                let ptr = unsafe { pool.ptr.add(current_offset) };

                // 2. OpenCL Sub-Buffer 생성
                // (실제 할당이 아니라 뷰만 생성하므로 매우 빠름)
                let sub_buffer = pool
                    .buffer
                    .create_sub_buffer(Some(flags::MEM_READ_WRITE), current_offset, count)
                    .unwrap();

                return Tensor::from_storage(
                    Storage::Shared {
                        ptr: ptr as *mut u8,
                        size: count * 4,
                        cl_buffer: Some(sub_buffer),
                        queue: Some(self.context.queue().clone()),
                    },
                    shape.clone(),
                    Device::OpenCl,
                );
            } else {
                // 풀이 가득 차면 경고 후 일반 할당으로 넘어감 (혹은 패닉)
                // eprintln!("[Warn] Scratch pool exhausted! Falling back to slow allocation.");
                error!("Scratch pool exhausted! Falling back to slow allocation.");
            }
        }

        // [Fallback] 풀이 없거나 가득 찬 경우 (기존 방식: 느림)
        let buffer = Buffer::<f32>::builder()
            .queue(self.context.queue().clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_ALLOC_HOST_PTR)
            .len(count)
            .build()
            .unwrap();

        let ptr = unsafe { buffer.map().enq().unwrap().as_mut_ptr() };

        Tensor::from_storage(
            Storage::Shared {
                ptr: ptr as *mut u8,
                size: count * 4,
                cl_buffer: Some(buffer),
                queue: Some(self.context.queue().clone()),
            },
            shape.clone(),
            Device::OpenCl,
        )
    }

    pub fn allocate_shared_q4(&self, shape: &Shape, data_len: usize, scale_len: usize) -> Tensor {
        let total_bytes = data_len + (scale_len * 4);
        let buffer = Buffer::<u8>::builder()
            .queue(self.context.queue().clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_ALLOC_HOST_PTR)
            .len(total_bytes)
            .build()
            .unwrap();

        let ptr = unsafe { buffer.map().enq().unwrap().as_mut_ptr() };

        Tensor::from_storage(
            Storage::SharedQ4 {
                ptr,
                size: total_bytes,
                cl_buffer: Some(buffer),
                queue: Some(self.context.queue().clone()),
                data_len,
                scale_len,
            },
            shape.clone(),
            Device::OpenCl,
        )
    }

    // --- Operations ---
    // Standard MatMul: A[M,K] x B[K,N] -> C[M,N]
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let (m, k) = (a.shape().dims()[0], a.shape().dims()[1]);
        let (k_b, n) = (b.shape().dims()[0], b.shape().dims()[1]);

        // Correct assertion for standard MatMul
        assert_eq!(
            k, k_b,
            "MatMul dimension mismatch: A cols {} != B rows {}",
            k, k_b
        );

        let c = self.allocate_shared(&Shape::new(vec![m, n]));

        // Usually 'b' in standard MatMul is an activation or unquantized tensor.
        // So we use F32 Standard Kernel.
        let kernel = self
            .context
            .kernel_builder("kernel_matmul_f32")
            .arg(self.get_buffer(a))
            .arg(self.get_buffer(b))
            .arg(self.get_buffer(&c))
            .arg(m as i32)
            .arg(k as i32)
            .arg(n as i32)
            .global_work_size([m, n])
            .build()
            .unwrap();
        unsafe {
            kernel.enq().unwrap();
        }

        self.context.queue().finish().unwrap();
        c
    }

    // Transposed MatMul: A[M,K] x B[N,K]^T -> C[M,N]
    // (Used for Linear Layers where weights B are stored as [Output, Input])
    pub fn matmul_transposed(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let (m, k) = (a.shape().dims()[0], a.shape().dims()[1]);
        let (n, k_b) = (b.shape().dims()[0], b.shape().dims()[1]);

        // Correct assertion for Transposed MatMul
        assert_eq!(
            k, k_b,
            "MatMul Transposed dimension mismatch: A cols {} != B cols {}",
            k, k_b
        );

        let c = self.allocate_shared(&Shape::new(vec![m, n]));

        match (&*a.storage, &*b.storage) {
            // Optimization for Q4 Weights (B is Q4 [N, K])
            (
                Storage::Shared { .. },
                Storage::SharedQ4 {
                    cl_buffer,
                    data_len,
                    scale_len,
                    ..
                },
            ) => {
                let cl_buffer = cl_buffer.as_ref().unwrap();

                if m == 1 {
                    // Decoding optimization
                    let buf_a = self.get_buffer(a);
                    let buf_c = self.get_buffer(&c);

                    let scale_offset = *data_len;
                    let buf_scales = cl_buffer
                        .create_sub_buffer(Some(flags::MEM_READ_WRITE), scale_offset, scale_len * 4)
                        .unwrap();

                    let kernel = self
                        .context
                        .kernel_builder("kernel_mul_mv_q4_0_f32")
                        .arg(cl_buffer) // qs
                        .arg(&buf_scales) // scales
                        .arg(buf_a) // vec input
                        .arg(buf_c) // vec output
                        .arg(k as i32)
                        .arg(n as i32)
                        .global_work_size(n)
                        .build()
                        .unwrap();

                    unsafe {
                        kernel.enq().unwrap();
                    }
                } else {
                    panic!("Q4 MatMul for batch > 1 not implemented yet");
                }
            }
            // Fallback for F32 x F32 Transposed
            _ => {
                let kernel = self
                    .context
                    .kernel_builder("kernel_matmul_f32_transposed")
                    .arg(self.get_buffer(a))
                    .arg(self.get_buffer(b))
                    .arg(self.get_buffer(&c))
                    .arg(m as i32)
                    .arg(k as i32)
                    .arg(n as i32)
                    .global_work_size([m, n])
                    .build()
                    .unwrap();
                unsafe {
                    kernel.enq().unwrap();
                }
            }
        }
        self.context.queue().finish().unwrap();
        c
    }

    // [Performance Critical Update]
    // Existing implementation of matmul_slice was CPU fallback.
    // We override it to use GPU kernel (F32 x F32).
    pub fn matmul_slice(&self, a: &Tensor, d: &[f32], r: usize, c: usize) -> Tensor {
        // a: Activation [Seq, Hidden] (F32)
        // d: Other data Slice (likely K cache transposed) [Rows, Cols] -> [r, c]

        let dims_a = a.shape().dims();
        let (m, k) = (dims_a[0], dims_a[1]);
        assert_eq!(k, c, "Slice dimension mismatch");

        // 1. Upload Slice 'd' to GPU (Temporary Tensor)
        let slice_shape = Shape::new(vec![r, c]);
        let slice_tensor = self.allocate_shared(&slice_shape);

        // Copy data to the mapped pointer
        unsafe {
            let ptr = match &*slice_tensor.storage {
                Storage::Shared { ptr, .. } => *ptr as *mut f32,
                _ => unreachable!(),
            };
            std::ptr::copy_nonoverlapping(d.as_ptr(), ptr, d.len());
        }

        // 2. Perform MatMul on GPU using the standard matmul (F32 x F32 path)
        self.matmul(a, &slice_tensor)
    }

    pub fn rms_norm(&self, x: &Tensor, w: &Tensor, eps: f32) -> Tensor {
        let count = x.shape().num_elements();
        let hidden = x.shape().dims().last().copied().unwrap();
        let rows = count / hidden;

        let o = self.allocate_shared(x.shape());
        let local_size = 256;

        let kernel = self
            .context
            .kernel_builder("kernel_rms_norm")
            .arg(self.get_buffer(x))
            .arg(self.get_buffer(&o))
            .arg(hidden as i32)
            .arg(eps)
            .arg_local::<f32>(local_size)
            .global_work_size(rows * local_size)
            .local_work_size(local_size)
            .build()
            .unwrap();

        unsafe {
            kernel.enq().unwrap();
        }

        // Note: Missing weight mul step here. Assuming kernel handles norm-only.

        self.context.queue().finish().unwrap();
        o
    }

    pub fn rope_inplace(&self, x: &mut Tensor, start_pos: usize) {
        let dims = x.shape().dims();
        let head_dim = *dims.last().unwrap();
        let n_heads = 1;
        let seq_len = dims[0];

        let kernel = self
            .context
            .kernel_builder("kernel_rope")
            .arg(self.get_buffer(x))
            .arg(head_dim as i32)
            .arg((head_dim / 2) as i32)
            .arg(n_heads as i32)
            .arg(start_pos as i32)
            .arg(10000.0f32) // freq_base
            .arg(1.0f32) // freq_scale
            .global_work_size([head_dim / 2, n_heads, seq_len])
            .build()
            .unwrap();

        unsafe {
            kernel.enq().unwrap();
        }
        self.context.queue().finish().unwrap();
    }

    pub fn silu_mul(&self, gate: &mut Tensor, up: &Tensor) {
        let n = gate.shape().num_elements();
        let kernel = self
            .context
            .kernel_builder("kernel_silu_mul")
            .arg(self.get_buffer(gate))
            .arg(self.get_buffer(up))
            .arg(n as i32)
            .global_work_size(n)
            .build()
            .unwrap();
        unsafe {
            kernel.enq().unwrap();
        }
        self.context.queue().finish().unwrap();
    }

    pub fn add_assign(&self, a: &mut Tensor, b: &Tensor) {
        let n = a.shape().num_elements();
        let kernel = self
            .context
            .kernel_builder("kernel_add")
            .arg(self.get_buffer(a))
            .arg(self.get_buffer(b))
            .arg(n as i32)
            .global_work_size(n)
            .build()
            .unwrap();
        unsafe {
            kernel.enq().unwrap();
        }
        self.context.queue().finish().unwrap();
    }

    pub fn softmax(&self, x: &Tensor) -> Tensor {
        let rows = x.shape().dims()[0];
        let cols = x.shape().dims()[1];
        let o = self.allocate_shared(x.shape());

        let kernel = self
            .context
            .kernel_builder("kernel_softmax_simple")
            .arg(self.get_buffer(x))
            .arg(self.get_buffer(&o))
            .arg(cols as i32)
            .global_work_size(rows)
            .build()
            .unwrap();

        unsafe {
            kernel.enq().unwrap();
        }
        self.context.queue().finish().unwrap();
        o
    }

    pub fn scale(&self, x: &Tensor, v: f32) -> Tensor {
        let mut o = self.allocate_shared(x.shape());
        // Placeholder: CPU implementation or add scale kernel
        unsafe {
            let src = match &*x.storage {
                Storage::Shared { ptr, .. } => *ptr as *const f32,
                _ => panic!("Invalid storage"),
            };
            let dst = match &*o.storage {
                Storage::Shared { ptr, .. } => *ptr as *mut f32,
                _ => unreachable!(),
            };
            let n = x.shape().num_elements();
            for i in 0..n {
                *dst.add(i) = *src.add(i) * v;
            }
        }
        o
    }

    pub fn copy_from(&self, t: &Tensor) -> Tensor {
        let mut o = self.allocate_shared(t.shape());
        unsafe {
            let src = match &*t.storage {
                Storage::Shared { ptr, .. } => *ptr,
                Storage::Cpu(d) => d.as_ptr() as *mut u8,
                _ => panic!("Clone source invalid"),
            };
            let dst = match &*o.storage {
                Storage::Shared { ptr, .. } => *ptr,
                _ => unreachable!(),
            };
            std::ptr::copy_nonoverlapping(src, dst, t.shape().num_elements() * 4);
        }
        o
    }

    pub fn launch_dummy_kernel(&self, t: &Tensor) {
        let n = t.shape().num_elements();
        // elementwise.cl 등에 간단한 덧셈 커널이 있다고 가정하거나
        // 기존의 kernel_add를 재사용할 수 있습니다.
        let kernel = self
            .context
            .kernel_builder("kernel_add") // 이미 구현된 커널 사용
            .arg(self.get_buffer(t))
            .arg(self.get_buffer(t)) // 자기 자신을 두 번 전달하거나 1.0을 더하는 커널 필요
            .arg(n as i32)
            .global_work_size(n)
            .build()
            .unwrap();

        unsafe {
            kernel.enq().unwrap();
        }
        self.context.queue().finish().unwrap();
    }
}

impl Backend for OpenClBackend {
    fn device(&self) -> Device {
        Device::OpenCl
    }
    fn name(&self) -> &str {
        "OpenCL (Kernel Files)"
    }
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        self.matmul(a, b)
    }
    fn matmul_transposed(&self, a: &Tensor, b: &Tensor) -> Tensor {
        self.matmul_transposed(a, b)
    }
    fn matmul_slice(&self, a: &Tensor, d: &[f32], r: usize, c: usize) -> Tensor {
        self.matmul_slice(a, d, r, c)
    }
    fn add_assign(&self, a: &mut Tensor, b: &Tensor) {
        self.add_assign(a, b)
    }
    fn silu_mul(&self, g: &mut Tensor, u: &Tensor) {
        self.silu_mul(g, u)
    }
    fn rope_inplace(&self, x: &mut Tensor, s: usize) {
        self.rope_inplace(x, s)
    }
    fn rms_norm(&self, x: &Tensor, w: &Tensor, e: f32) -> Tensor {
        self.rms_norm(x, w, e)
    }
    fn softmax(&self, x: &Tensor) -> Tensor {
        self.softmax(x)
    }
    fn scale(&self, x: &Tensor, v: f32) -> Tensor {
        self.scale(x, v)
    }
    fn copy_from(&self, t: &Tensor) -> Tensor {
        self.copy_from(t)
    }
}
