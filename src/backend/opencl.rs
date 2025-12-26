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
            // [Step 2.5] Tiled Q4 Kernel 적용 (성능 개선의 핵심)
            let src = r#"
                // Tile Size (32x32 블록 사용)
                #define TS 32

                __kernel void matmul_tiled_f32(
                    __global const float* A,
                    __global const float* B,
                    __global float* C,
                    const int M, const int K, const int N)
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
                        const int tiledRow = TS * t + row;
                        
                        // Load Tiles
                        Asub[row][col] = A[globalRow * K + tiledCol];
                        // B is Transposed [N, K] logically
                        Bsub[row][col] = B[globalCol * K + tiledRow]; 

                        barrier(CLK_LOCAL_MEM_FENCE);

                        // Compute Block
                        for (int k = 0; k < TS; k++) {
                            acc += Asub[row][k] * Bsub[col][k];
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (globalRow < M && globalCol < N) C[globalRow * N + globalCol] = acc;
                }

                // [핵심] Tiled Q4 MatMul
                // A: [M, K] (F32)
                // B: [N, K] (Q4 Packed + Scales) -> We use Shared Memory to cache unpacked F32s
                __kernel void matmul_q4_tiled(
                    __global const float* A,
                    __global const uchar* B_data,
                    __global float* C,
                    const int M, 
                    const int K, 
                    const int N,
                    const int offset_scales)
                {
                    // TS=32, WorkGroup=32x32 recommended (or optimized mapping)
                    const int local_y = get_local_id(0); // 0..31 (Rows of A / C)
                    const int local_x = get_local_id(1); // 0..31 (Cols of B / C)
                    
                    const int global_y = get_group_id(0) * TS + local_y; // Global Row (M)
                    const int global_x = get_group_id(1) * TS + local_x; // Global Col (N)

                    // Local Memory Cache
                    __local float Asub[TS][TS];
                    __local float Bsub[TS][TS];

                    float acc = 0.0f;
                    const int num_tiles = K / TS;
                    
                    // Scales start pointer
                    __global const float* B_scales = (__global const float*)(B_data + offset_scales);

                    for (int t = 0; t < num_tiles; t++) {
                        // 1. Load A Tile (F32) -> Simple Copy
                        // A[global_y][t*TS + local_x]
                        const int tiled_k_a = t * TS + local_x;
                        Asub[local_y][local_x] = A[global_y * K + tiled_k_a];

                        // 2. Load B Tile (Q4) -> Dequantize -> Store F32
                        // We need B[global_x][t*TS + local_y] (Since B is [N, K])
                        // Each thread loads one pixel of Bsub.
                        const int tiled_k_b = t * TS + local_y; // k index for B
                        
                        // Calculate Q4 Indices
                        // Block Index = k / 32. Since TS=32, 't' is exactly the block index offset?
                        // Actually, k goes 0..K. Block size is 32.
                        // tiled_k_b / 32 determines the scale.
                        
                        int b_row_offset = global_x * (K / 2); // Byte offset for row N
                        int k_div_2 = tiled_k_b / 2;
                        uchar packed = B_data[b_row_offset + k_div_2];
                        
                        // Extract Nibble (Even k = Low, Odd k = High)
                        // This branching is minor compared to memory latency
                        float nibble = (tiled_k_b % 2 == 0) ? (float)(packed & 0x0F) : (float)(packed >> 4);
                        
                        // Scale
                        // Scale Stride = K / 32.
                        // Scale Index = global_x * (K/32) + (tiled_k_b / 32)
                        float scale = B_scales[global_x * (K / 32) + (tiled_k_b / 32)];
                        
                        // Dequantize & Store to Local
                        Bsub[local_x][local_y] = (nibble - 8.0f) * scale;

                        // Synchronize to ensure all data is loaded
                        barrier(CLK_LOCAL_MEM_FENCE);

                        // 3. Compute (Dot Product of loaded tiles)
                        // A_row is Asub[local_y][:]
                        // B_col is Bsub[local_x][:] (Since we stored B transposed in shared mem for coalesced access?
                        // Wait, we stored Bsub[local_x][local_y] = B_val.
                        // B_val was B[global_x][k].
                        // So Bsub[local_x] contains the column-strip of the original matrix B^T?
                        // Let's trace:
                        // acc += A[row][k] * B[col][k]
                        // -> acc += Asub[local_y][k] * B_val_at_col_k
                        // B_val_at_col_k was B[global_x][current_k]
                        // loaded by thread where local_x implies global_x, and local_y implies k.
                        // So Bsub[local_x][local_y] holds B[global_x][t*TS + local_y].
                        // So to iterate k (0..31):
                        // A value is Asub[local_y][k]
                        // B value is Bsub[local_x][k]
                        
                        for (int k = 0; k < TS; k++) {
                            acc += Asub[local_y][k] * Bsub[local_x][k];
                        }
                        
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }

                    if (global_y < M && global_x < N) {
                        C[global_y * N + global_x] = acc;
                    }
                }
                
                // (기타 Helper 커널들 유지...)
                __kernel void add_dummy(__global float* a) { a[get_global_id(0)] += 1.0f; }
                __kernel void add_assign(__global float* a, __global const float* b, const int n) {
                    int i = get_global_id(0); if (i < n) a[i] += b[i];
                }
                __kernel void silu_mul(__global float* g, __global const float* u, const int n) {
                    int i = get_global_id(0); if (i < n) { float v = g[i]; g[i] = (v/(1.0f+exp(-v)))*u[i]; }
                }
                __kernel void scale(__global float* x, const float v, const int n) {
                    int i = get_global_id(0); if (i < n) x[i] *= v;
                }
                __kernel void rms_norm(__global const float* x, __global const float* w, __global float* o, const int s, const float e) {
                    int r = get_global_id(0); int off = r*s; float ss=0.0f;
                    for(int i=0;i<s;i++) ss+=x[off+i]*x[off+i];
                    float rms=sqrt(ss/s+e); float inv=1.0f/rms;
                    for(int i=0;i<s;i++) o[off+i]=x[off+i]*inv*w[i];
                }
                __kernel void softmax(__global const float* x, __global float* o, const int s) {
                    int r = get_global_id(0); int off = r*s; float max_v=-1e30f;
                    for(int i=0;i<s;i++) if(x[off+i]>max_v) max_v=x[off+i];
                    float sum=0.0f;
                    for(int i=0;i<s;i++) { float v=exp(x[off+i]-max_v); o[off+i]=v; sum+=v; }
                    float inv=1.0f/sum; for(int i=0;i<s;i++) o[off+i]*=inv;
                }
                __kernel void rope_inplace(__global float* x, const int s, const int d, const float b) {
                    int id=get_global_id(0); int m=d/2; int r=id/m; int j=id%m;
                    int p=s+r; float f=((float)j*2.0f)/d; float th=1.0f/pow(b,f);
                    float mth=p*th; float sn=sin(mth); float cs=cos(mth);
                    int idx=r*d+j; float re=x[idx]; float im=x[idx+m];
                    x[idx]=re*cs-im*sn; x[idx+m]=re*sn+im*cs;
                }
            "#;

            let pro_que = ProQue::builder()
                .src(src)
                .dims(1)
                .build()
                .expect("Failed to initialize OpenCL");

            println!("[OpenCL] Tiled Q4 Kernel Loaded (TS=32)");
            Arc::new(pro_que)
        });

        Self {
            context: context.clone(),
        }
    }

    // ... (allocate_shared_q4, allocate_shared, get_buffer 등은 기존과 동일) ...
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
    pub fn allocate_shared(&self, shape: &Shape) -> Tensor {
        let count = shape.num_elements();
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
    fn get_buffer<'a>(&self, t: &'a Tensor) -> &'a Buffer<f32> {
        match &*t.storage {
            Storage::Shared {
                cl_buffer: Some(b), ..
            } => b,
            _ => panic!("Not F32"),
        }
    }

    // [Step 2.5] Dispatcher Update
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
            // F32 x F32
            (Storage::Shared { .. }, Storage::Shared { .. }) => {
                let name = "matmul_tiled_f32";
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
                    .global_work_size([((m + 31) / 32) * 32, ((n + 31) / 32) * 32])
                    .local_work_size([32, 32]) // TS=32
                    .build()
                    .unwrap();
                unsafe {
                    kernel.enq().unwrap();
                }
            }

            // F32 x Q4 (Optimized Tiled)
            (
                Storage::Shared { .. },
                Storage::SharedQ4 {
                    cl_buffer,
                    data_len,
                    ..
                },
            ) => {
                if !transpose_b {
                    panic!("Q4 Tiled requires transposed B");
                }

                let buf_a = self.get_buffer(a);
                let buf_b_raw = cl_buffer.as_ref().unwrap();
                let buf_c = self.get_buffer(&c);

                // TS=32, WorkGroup=32x32 (1024 threads)
                // 만약 GPU가 1024 스레드를 지원하지 않으면 16x16으로 줄여야 함.
                // 대부분의 현대 GPU(ARM Mali G710+, Adreno 7xx, Apple M-series)는 지원.
                let kernel = self
                    .context
                    .kernel_builder("matmul_q4_tiled")
                    .arg(buf_a)
                    .arg(buf_b_raw)
                    .arg(buf_c)
                    .arg(m as i32)
                    .arg(k as i32)
                    .arg(n as i32)
                    .arg(*data_len as i32)
                    .global_work_size([((m + 31) / 32) * 32, ((n + 31) / 32) * 32])
                    .local_work_size([32, 32])
                    .build()
                    .expect("Build Q4 Tiled Failed");

                unsafe {
                    kernel.enq().expect("Enq Q4 Tiled Failed");
                }
            }
            _ => panic!("Storage mismatch"),
        }
        self.context.queue().finish().unwrap();
        c
    }

    // --- (나머지 Slice, Ops 등은 동일하게 유지) ---
    pub fn matmul_slice(&self, a: &Tensor, d: &[f32], r: usize, c: usize) -> Tensor {
        let dims_a = a.shape().dims();
        let (m, k) = (dims_a[0], dims_a[1]);
        let mut res = vec![0.0; m * r];
        let ad = a.data();
        for i in 0..m {
            let ar = &ad[i * k..(i + 1) * k];
            for j in 0..r {
                let br = &d[j * c..(j + 1) * c];
                res[i * r + j] = ar.iter().zip(br).map(|(x, y)| x * y).sum();
            }
        }
        let mut t = self.allocate_shared(&Shape::new(vec![m, r]));
        unsafe {
            std::ptr::copy_nonoverlapping(res.as_ptr(), t.data_mut().as_mut_ptr(), res.len());
        }
        t
    }
    fn add_assign(&self, a: &mut Tensor, b: &Tensor) {
        let n = a.shape().num_elements();
        let k = self
            .context
            .kernel_builder("add_assign")
            .arg(self.get_buffer(a))
            .arg(self.get_buffer(b))
            .arg(n as i32)
            .global_work_size(n)
            .build()
            .unwrap();
        unsafe {
            k.enq().unwrap();
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
    fn rope_inplace(&self, x: &mut Tensor, s: usize) {
        let dims = x.shape().dims();
        let hd = *dims.last().unwrap();
        let n = x.shape().num_elements() / 2;
        let k = self
            .context
            .kernel_builder("rope_inplace")
            .arg(self.get_buffer(x))
            .arg(s as i32)
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
    fn rms_norm(&self, x: &Tensor, w: &Tensor, e: f32) -> Tensor {
        let dims = x.shape().dims();
        let s = *dims.last().unwrap();
        let r = x.shape().num_elements() / s;
        let o = self.allocate_shared(x.shape());
        let k = self
            .context
            .kernel_builder("rms_norm")
            .arg(self.get_buffer(x))
            .arg(self.get_buffer(w))
            .arg(self.get_buffer(&o))
            .arg(s as i32)
            .arg(e)
            .global_work_size(r)
            .build()
            .unwrap();
        unsafe {
            k.enq().unwrap();
        }
        self.context.queue().finish().unwrap();
        o
    }
    fn softmax(&self, x: &Tensor) -> Tensor {
        let dims = x.shape().dims();
        let s = *dims.last().unwrap();
        let r = x.shape().num_elements() / s;
        let o = self.allocate_shared(x.shape());
        let k = self
            .context
            .kernel_builder("softmax")
            .arg(self.get_buffer(x))
            .arg(self.get_buffer(&o))
            .arg(s as i32)
            .global_work_size(r)
            .build()
            .unwrap();
        unsafe {
            k.enq().unwrap();
        }
        self.context.queue().finish().unwrap();
        o
    }
    fn scale(&self, x: &Tensor, v: f32) -> Tensor {
        let o = self.copy_from(x);
        let n = x.shape().num_elements();
        let k = self
            .context
            .kernel_builder("scale")
            .arg(self.get_buffer(&o))
            .arg(v)
            .arg(n as i32)
            .global_work_size(n)
            .build()
            .unwrap();
        unsafe {
            k.enq().unwrap();
        }
        self.context.queue().finish().unwrap();
        o
    }
    fn copy_from(&self, t: &Tensor) -> Tensor {
        let mut o = self.allocate_shared(t.shape());
        unsafe {
            std::ptr::copy_nonoverlapping(
                t.data().as_ptr(),
                o.data_mut().as_mut_ptr(),
                t.shape().num_elements(),
            );
        }
        o
    }
    pub fn launch_dummy_kernel(&self, _: &Tensor) {}
}

impl Backend for OpenClBackend {
    fn device(&self) -> Device {
        Device::OpenCl
    }
    fn name(&self) -> &str {
        "ARM OpenCL (Tiled Q4)"
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
