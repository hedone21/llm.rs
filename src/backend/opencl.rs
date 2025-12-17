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
            // [검증용 커널] 배열의 모든 원소에 1.0을 더함
            let src = r#"
                __kernel void add_dummy(__global float* a) {
                    int id = get_global_id(0);
                    a[id] += 1.0f;
                }
            "#;

            let pro_que = ProQue::builder()
                .src(src)
                .dims(1) // 나중에 커널 실행 시 재설정
                .build()
                .expect("Failed to initialize OpenCL. Check generic ARM GPU drivers.");

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

        // [수정] 큐를 함께 저장
        Tensor::from_storage(
            Storage::Shared {
                ptr: ptr as *mut u8,
                size: size_bytes,
                cl_buffer: Some(buffer),
                queue: Some(self.context.queue().clone()), // Drop 시 Unmap을 위해 저장
                handle: 0,
            },
            shape.clone(),
            Device::OpenCl,
        )
    }

    // [신규] Phase 1 검증 함수: GPU에서 값 수정이 CPU에 즉시 반영되는지 확인
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

            // 명령 대기열 완료 대기 (동기화)
            self.context
                .queue()
                .finish()
                .expect("Failed to finish queue");
        } else {
            panic!("Tensor is not backed by Shared OpenCL Buffer");
        }
    }
}

// ... (Backend trait impl 부분은 기존과 동일)
impl Backend for OpenClBackend {
    fn device(&self) -> Device {
        Device::OpenCl
    }
    fn name(&self) -> &str {
        "ARM OpenCL (Zero-Copy)"
    }
    fn matmul(&self, _a: &Tensor, _b: &Tensor) -> Tensor {
        unimplemented!()
    }
    fn matmul_transposed(&self, _a: &Tensor, _b: &Tensor) -> Tensor {
        unimplemented!()
    }
    fn matmul_slice(&self, _a: &Tensor, _other: &[f32], _r: usize, _c: usize) -> Tensor {
        unimplemented!()
    }
    fn add_assign(&self, _a: &mut Tensor, _b: &Tensor) {
        unimplemented!()
    }
    fn silu_mul(&self, _gate: &mut Tensor, _up: &Tensor) {
        unimplemented!()
    }
    fn rope_inplace(&self, _x: &mut Tensor, _pos: usize) {
        unimplemented!()
    }
    fn rms_norm(&self, _x: &Tensor, _w: &Tensor, _e: f32) -> Tensor {
        unimplemented!()
    }
    fn softmax(&self, _x: &Tensor) -> Tensor {
        unimplemented!()
    }
    fn scale(&self, _x: &Tensor, _v: f32) -> Tensor {
        unimplemented!()
    }
    fn copy_from(&self, _t: &Tensor) -> Tensor {
        unimplemented!()
    }
}

