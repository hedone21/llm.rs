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
            // [Step 2 Fix] matmul_q4_simd 커널의 시프트 연산 타입 수정 ((uchar4)(4))
            let src = r#"
                #define TS 16

                __kernel void add_dummy(__global float* a) {
                    int id = get_global_id(0);
                    a[id] += 1.0f;
                }

                // [Optimized] Tiled MatMul for F32
                __kernel void matmul_tiled(
                    __global const float* A,
                    __global const float* B,
                    __global float* C,
                    const int M,
                    const int K,
                    const int N)
                {
                    const int row = get_local_id(0); 
                    const int col = get_local_id(1); 
                    const int globalRow = TS * get_group_id(0) + row; 
                    const int globalCol = TS * get_group_id(1) + col; 

                    __local float Asub[TS][TS];
                    __local float Bsub[TS][TS];

                    float acc = 0.0f;

                    const int numTiles = K / TS;
                    for (int t = 0; t < numTiles; t++) {
                        const int tiledRow = TS * t + row;
                        const int tiledCol = TS * t + col;
                        
                        Asub[row][col] = A[globalRow * K + tiledCol];
                        Bsub[row][col] = B[tiledRow * N + globalCol];

                        barrier(CLK_LOCAL_MEM_FENCE);

                        for (int k = 0; k < TS; k++) {
                            acc += Asub[row][k] * Bsub[k][col];
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    
                    if (globalRow < M && globalCol < N) {
                        C[globalRow * N + globalCol] = acc;
                    }
                }

                // [Optimized] Tiled MatMul Transposed (A @ B^T)
                __kernel void matmul_transposed_tiled(
                    __global const float* A,
                    __global const float* B,
                    __global float* C,
                    const int M,
                    const int K,
                    const int N)
                {
                    const int row = get_local_id(0);
                    const int col = get_local_id(1);
                    const int globalRow = TS * get_group_id(0) + row;
                    const int globalCol = TS * get_group_id(1) + col;

                    __local float Asub[TS][TS];
                    __local float Bsub[TS][TS];

                    float acc = 0.0f;

                    const int numTiles = K / TS;
                    for (int t = 0; t < numTiles; t++) {
                        const int tiledCol = TS * t + col; 
                        const int tiledColB = TS * t + row; 

                        Asub[row][col] = A[globalRow * K + (TS * t + col)];
                        Bsub[row][col] = B[globalCol * K + (TS * t + row)]; 

                        barrier(CLK_LOCAL_MEM_FENCE);

                        for (int k = 0; k < TS; k++) {
                            acc += Asub[row][k] * Bsub[col][k];
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }

                    if (globalRow < M && globalCol < N) {
                        C[globalRow * N + globalCol] = acc;
                    }
                }

                // [Optimized] Q4 MatMul (SIMD + Loop Unrolling)
                __kernel void matmul_q4_simd(
                    __global const float* A,
                    __global const uchar* B_data,
                    __global float* C,
                    const int M, 
                    const int K, 
                    const int N,
                    const int offset_scales)
                {
                    int row = get_global_id(0); 
                    int col = get_global_id(1); 

                    __global const float* B_scales = (__global const float*)(B_data + offset_scales);

                    if (row < M && col < N) {
                        float4 sum_vec = (float4)(0.0f); 
                        
                        int b_row_offset = col * (K / 2); 
                        int s_row_offset = col * (K / 32); 

                        __global const uchar* my_b_data = B_data + b_row_offset;
                        __global const float* my_b_scales = B_scales + s_row_offset;
                        __global const float* my_a_row = A + row * K;
                        
                        int num_blocks = K / 32;
                        
                        for (int b = 0; b < num_blocks; b++) {
                            float scale = my_b_scales[b];
                            
                            int base_idx = b * 16;    
                            int a_base_idx = b * 32;   

                            for (int i = 0; i < 4; i++) {
                                uchar4 packed = vload4(0, my_b_data + base_idx + i * 4);
                                
                                float4 a1 = vload4(0, my_a_row + a_base_idx + i * 8);
                                float4 a2 = vload4(0, my_a_row + a_base_idx + i * 8 + 4);

                                // Lower nibbles
                                float4 w1 = convert_float4(packed & (uchar4)(0x0F)) - 8.0f;
                                // Upper nibbles [FIX: Explicit cast for shift]
                                float4 w2 = convert_float4(packed >> (uchar4)(4)) - 8.0f;

                                sum_vec.x += a1.x * (w1.x * scale);
                                sum_vec.y += a1.y * (w2.x * scale);
                                
                                sum_vec.z += a1.z * (w1.y * scale);
                                sum_vec.w += a1.w * (w2.y * scale);
                                
                                sum_vec.x += a2.x * (w1.z * scale);
                                sum_vec.y += a2.y * (w2.z * scale);
                                
                                sum_vec.z += a2.z * (w1.w * scale);
                                sum_vec.w += a2.w * (w2.w * scale);
                            }
                        }
                        C[row * N + col] = sum_vec.x + sum_vec.y + sum_vec.z + sum_vec.w;
                    }
                }
                
                __kernel void add_assign(__global float* a, __global const float* b, const int n) {
                    int i = get_global_id(0); if (i < n) a[i] += b[i];
                }
                __kernel void silu_mul(__global float* gate, __global const float* up, const int n) {
                    int i = get_global_id(0);
                    if (i < n) {
                        float g = gate[i];
                        gate[i] = (g / (1.0f + exp(-g))) * up[i];
                    }
                }
                __kernel void scale(__global float* x, const float val, const int n) {
                    int i = get_global_id(0); if (i < n) x[i] *= val;
                }
                __kernel void rms_norm(__global const float* x, __global const float* w, __global float* out, const int stride, const float eps) {
                    int row = get_global_id(0);
                    int offset = row * stride;
                    float ss = 0.0f;
                    for (int i=0; i<stride; i++) ss += x[offset+i] * x[offset+i];
                    float rms = sqrt(ss / stride + eps);
                    float s = 1.0f / rms;
                    for (int i=0; i<stride; i++) out[offset+i] = x[offset+i] * s * w[i];
                }
                __kernel void softmax(__global const float* x, __global float* out, const int stride) {
                    int row = get_global_id(0);
                    int offset = row * stride;
                    float max_v = -1e30f;
                    for(int i=0; i<stride; i++) if(x[offset+i] > max_v) max_v = x[offset+i];
                    float sum = 0.0f;
                    for(int i=0; i<stride; i++) {
                        float e = exp(x[offset+i] - max_v);
                        out[offset+i] = e;
                        sum += e;
                    }
                    float inv = 1.0f / sum;
                    for(int i=0; i<stride; i++) out[offset+i] *= inv;
                }
                __kernel void rope_inplace(__global float* x, const int start, const int dim, const float base) {
                    int gid = get_global_id(0);
                    int mid = dim / 2;
                    int row = gid / mid;
                    int j = gid % mid;
                    int pos = start + row;
                    float freq = ((float)j * 2.0f) / (float)dim;
                    float theta = 1.0f / pow(base, freq);
                    float m_theta = (float)pos * theta;
                    float s = sin(m_theta);
                    float c = cos(m_theta);
                    int idx = row * dim + j;
                    float re = x[idx];
                    float im = x[idx + mid];
                    x[idx] = re * c - im * s;
                    x[idx + mid] = re * s + im * c;
                }
            "#;

            let pro_que = ProQue::builder()
                .src(src)
                .dims(1)
                .build()
                .expect("Failed to initialize OpenCL");

            println!("[OpenCL] Optimized Kernels Loaded (TS=16, SIMD)");
            Arc::new(pro_que)
        });

        Self {
            context: context.clone(),
        }
    }

    pub fn allocate_shared_q4(&self, shape: &Shape, data_len: usize, scale_len: usize) -> Tensor {
        let total_bytes = data_len + (scale_len * 4);
        let buffer = Buffer::<u8>::builder()
            .queue(self.context.queue().clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_ALLOC_HOST_PTR)
            .len(total_bytes)
            .build()
            .expect("Q4 Alloc Failed");
        let ptr = unsafe { buffer.map().enq().expect("Map Failed").as_mut_ptr() };
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

    pub fn allocate_shared(&self, shape: &Shape) -> Tensor {
        let count = shape.num_elements();
        let buffer = Buffer::<f32>::builder()
            .queue(self.context.queue().clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_ALLOC_HOST_PTR)
            .len(count)
            .build()
            .expect("Alloc Failed");
        let ptr = unsafe { buffer.map().enq().expect("Map Failed").as_mut_ptr() };
        Tensor::from_storage(
            Storage::Shared {
                ptr: ptr as *mut u8,
                size: count * 4,
                cl_buffer: Some(buffer),
                queue: Some(self.context.queue().clone()),
                handle: 0,
            },
            shape.clone(),
            Device::OpenCl,
        )
    }

    fn get_buffer<'a>(&self, tensor: &'a Tensor) -> &'a Buffer<f32> {
        match &*tensor.storage {
            Storage::Shared {
                cl_buffer: Some(b), ..
            } => b,
            Storage::OpenCl(b) => b,
            _ => panic!("Not OpenCL F32 Tensor"),
        }
    }

    fn matmul_dispatch(&self, a: &Tensor, b: &Tensor, transpose_b: bool) -> Tensor {
        let dim_a = a.shape().dims();
        let dim_b = b.shape().dims();
        let (m, k) = (dim_a[0], dim_a[1]);
        let (n, k2) = if transpose_b {
            (dim_b[0], dim_b[1])
        } else {
            (dim_b[1], dim_b[0])
        };
        assert_eq!(k, k2, "Dim mismatch");

        let c = self.allocate_shared(&Shape::new(vec![m, n]));

        match (&*a.storage, &*b.storage) {
            (Storage::Shared { .. }, Storage::Shared { .. })
            | (Storage::OpenCl(_), Storage::OpenCl(_)) => {
                let name = if transpose_b {
                    "matmul_transposed_tiled"
                } else {
                    "matmul_tiled"
                };
                let buf_a = self.get_buffer(a);
                let buf_b = self.get_buffer(b);
                let buf_c = self.get_buffer(&c);

                let kernel = self
                    .context
                    .kernel_builder(name)
                    .arg(buf_a)
                    .arg(buf_b)
                    .arg(buf_c)
                    .arg(m as i32)
                    .arg(k as i32)
                    .arg(n as i32)
                    .global_work_size([((m + 15) / 16) * 16, ((n + 15) / 16) * 16])
                    .local_work_size([16, 16])
                    .build()
                    .unwrap();

                unsafe {
                    kernel.enq().unwrap();
                }
                self.context.queue().finish().unwrap();
            }

            (
                Storage::Shared { .. },
                Storage::SharedQ4 {
                    cl_buffer,
                    data_len,
                    ..
                },
            ) => {
                if !transpose_b {
                    panic!("Q4 only supports transposed B");
                }

                let buf_a = self.get_buffer(a);
                let buf_b_raw = cl_buffer.as_ref().unwrap();
                let buf_c = self.get_buffer(&c);

                let kernel = self
                    .context
                    .kernel_builder("matmul_q4_simd")
                    .arg(buf_a)
                    .arg(buf_b_raw)
                    .arg(buf_c)
                    .arg(m as i32)
                    .arg(k as i32)
                    .arg(n as i32)
                    .arg(*data_len as i32)
                    .global_work_size([m, n])
                    .build()
                    .expect("Build Q4 SIMD failed");

                unsafe {
                    kernel.enq().expect("Enq Q4 SIMD failed");
                }
                self.context
                    .queue()
                    .finish()
                    .expect("Finish Q4 SIMD failed");
            }
            _ => panic!("Storage mismatch"),
        }
        c
    }

    pub fn matmul_slice(&self, a: &Tensor, other_data: &[f32], rows: usize, cols: usize) -> Tensor {
        let dims_a = a.shape().dims();
        let (m, k) = (dims_a[0], dims_a[1]);
        let mut result_data = vec![0.0; m * rows];
        let a_data = a.data();
        for i in 0..m {
            let a_row = &a_data[i * k..(i + 1) * k];
            for j in 0..rows {
                let b_row = &other_data[j * cols..(j + 1) * cols];
                let sum: f32 = a_row.iter().zip(b_row.iter()).map(|(x, y)| x * y).sum();
                result_data[i * rows + j] = sum;
            }
        }
        let mut res = self.allocate_shared(&Shape::new(vec![m, rows]));
        unsafe {
            std::ptr::copy_nonoverlapping(
                result_data.as_ptr(),
                res.data_mut().as_mut_ptr(),
                result_data.len(),
            );
        }
        res
    }

    fn add_assign(&self, a: &mut Tensor, b: &Tensor) {
        let n = a.shape().num_elements();
        let ka = self
            .context
            .kernel_builder("add_assign")
            .arg(self.get_buffer(a))
            .arg(self.get_buffer(b))
            .arg(n as i32)
            .global_work_size(n)
            .build()
            .unwrap();
        unsafe {
            ka.enq().unwrap();
        }
        self.context.queue().finish().unwrap();
    }
    fn silu_mul(&self, g: &mut Tensor, u: &Tensor) {
        let n = g.shape().num_elements();
        let k = self
            .context
            .kernel_builder("silu_mul")
            .arg(self.get_buffer(g))
            .arg(self.get_buffer(u))
            .arg(n as i32)
            .global_work_size(n)
            .build()
            .unwrap();
        unsafe {
            k.enq().unwrap();
        }
        self.context.queue().finish().unwrap();
    }
    fn rope_inplace(&self, x: &mut Tensor, start: usize) {
        let dims = x.shape().dims();
        let hd = *dims.last().unwrap();
        let n = x.shape().num_elements() / 2;
        let k = self
            .context
            .kernel_builder("rope_inplace")
            .arg(self.get_buffer(x))
            .arg(start as i32)
            .arg(hd as i32)
            .arg(500000.0f32)
            .global_work_size(n)
            .build()
            .unwrap();
        unsafe {
            k.enq().unwrap();
        }
        self.context.queue().finish().unwrap();
    }
    fn rms_norm(&self, x: &Tensor, w: &Tensor, eps: f32) -> Tensor {
        let dims = x.shape().dims();
        let stride = *dims.last().unwrap();
        let rows = x.shape().num_elements() / stride;
        let out = self.allocate_shared(x.shape());
        let k = self
            .context
            .kernel_builder("rms_norm")
            .arg(self.get_buffer(x))
            .arg(self.get_buffer(w))
            .arg(self.get_buffer(&out))
            .arg(stride as i32)
            .arg(eps)
            .global_work_size(rows)
            .build()
            .unwrap();
        unsafe {
            k.enq().unwrap();
        }
        self.context.queue().finish().unwrap();
        out
    }
    fn softmax(&self, x: &Tensor) -> Tensor {
        let dims = x.shape().dims();
        let stride = *dims.last().unwrap();
        let rows = x.shape().num_elements() / stride;
        let out = self.allocate_shared(x.shape());
        let k = self
            .context
            .kernel_builder("softmax")
            .arg(self.get_buffer(x))
            .arg(self.get_buffer(&out))
            .arg(stride as i32)
            .global_work_size(rows)
            .build()
            .unwrap();
        unsafe {
            k.enq().unwrap();
        }
        self.context.queue().finish().unwrap();
        out
    }
    fn scale(&self, x: &Tensor, v: f32) -> Tensor {
        let out = self.copy_from(x);
        let n = x.shape().num_elements();
        let k = self
            .context
            .kernel_builder("scale")
            .arg(self.get_buffer(&out))
            .arg(v)
            .arg(n as i32)
            .global_work_size(n)
            .build()
            .unwrap();
        unsafe {
            k.enq().unwrap();
        }
        self.context.queue().finish().unwrap();
        out
    }
    fn copy_from(&self, t: &Tensor) -> Tensor {
        let mut out = self.allocate_shared(t.shape());
        unsafe {
            std::ptr::copy_nonoverlapping(
                t.data().as_ptr(),
                out.data_mut().as_mut_ptr(),
                t.shape().num_elements(),
            );
        }
        out
    }
    pub fn launch_dummy_kernel(&self, t: &Tensor) {}
}

impl Backend for OpenClBackend {
    fn device(&self) -> Device {
        Device::OpenCl
    }
    fn name(&self) -> &str {
        "ARM OpenCL (Tiled+SIMD)"
    }
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        self.matmul_dispatch(a, b, false)
    }
    fn matmul_transposed(&self, a: &Tensor, b: &Tensor) -> Tensor {
        self.matmul_dispatch(a, b, true)
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
