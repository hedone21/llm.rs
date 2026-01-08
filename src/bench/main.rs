use clap::Parser;
use llmrs::backend::{Backend, Device};
use llmrs::core::shape::Shape;
use llmrs::core::tensor::Tensor;
use log::info;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of iterations for measurement
    #[arg(short, long, default_value_t = 10)]
    iterations: usize,

    /// Number of warmup iterations
    #[arg(short, long, default_value_t = 3)]
    warmup: usize,
}

fn main() {
    // env_logger::init(); // Optional if we want log output
    let args = Args::parse();

    println!(
        "{:<25} | {:<20} | {:<15} | {:<15} | {:<10}",
        "Operation", "Size", "CPU (ms)", "OpenCL (ms)", "Speedup"
    );
    println!(
        "{:-<25}-|-{:-<20}-|-{:-<15}-|-{:-<15}-|-{:-<10}",
        "", "", "", "", ""
    );

    let square_sizes = vec![128, 256, 512, 1024];

    // 1. MatMul (A @ B)
    for &n in &square_sizes {
        bench_matmul(n, n, n, &args);
    }

    // 2. MatMul Transposed (A @ B^T)
    for &n in &square_sizes {
        bench_matmul_transposed(n, n, n, &args);
    }

    // 3. MatMul Slice (Scaling K, typical decoding usage)
    let decoding_sizes = vec![
        (1, 4096, 128), // 1 token, 4096 context, 128 head_dim -> A=[1, 128], Slice=[4096, 128]
        (1, 4096, 4096), // 1 token, 4096 context, 4096 hidden
    ];
    for (m, rows, cols) in decoding_sizes {
        bench_matmul_slice(m, rows, cols, &args);
    }

    // 4. RMS Norm
    for &n in &square_sizes {
        bench_rms_norm(1, n, &args); // Batch 1
        bench_rms_norm(128, n, &args); // Batch 128
    }

    // 5. Rope
    for &n in &square_sizes {
        bench_rope(1, 128, n, &args); // Batch 1, Head 128, Seq N
    }

    // 6. Softmax
    for &n in &square_sizes {
        bench_softmax(128, n, &args);
    }
}

fn bench_matmul(m: usize, k: usize, n: usize, args: &Args) {
    let shape_a = Shape::new(vec![m, k]);
    let shape_b = Shape::new(vec![k, n]);

    let a_cpu = Tensor::new(vec![0.5; m * k], shape_a.clone());
    let b_cpu = Tensor::new(vec![0.5; k * n], shape_b.clone());

    // CPU
    let cpu_time = measure(args.warmup, args.iterations, || {
        let _c = a_cpu.matmul(&b_cpu);
    });

    #[cfg(feature = "opencl")]
    {
        let a_cl = a_cpu.to_device(Device::OpenCl);
        let b_cl = b_cpu.to_device(Device::OpenCl);

        // OpenCL
        let cl_time = measure_opencl(args.warmup, args.iterations, || {
            let _c = a_cl.matmul(&b_cl);
        });

        print_result(
            format!("MatMul [{},{},{}]", m, k, n).as_str(),
            format!("M={} K={} N={}", m, k, n).as_str(),
            cpu_time,
            cl_time,
        );
    }
}

fn bench_matmul_transposed(m: usize, k: usize, n: usize, args: &Args) {
    let shape_a = Shape::new(vec![m, k]);
    let shape_b = Shape::new(vec![n, k]); // Transposed B: [N, K]

    let a_cpu = Tensor::new(vec![0.5; m * k], shape_a.clone());
    let b_cpu = Tensor::new(vec![0.5; n * k], shape_b.clone());

    // CPU
    let cpu_time = measure(args.warmup, args.iterations, || {
        let _c = a_cpu.matmul_transposed(&b_cpu);
    });

    #[cfg(feature = "opencl")]
    {
        let a_cl = a_cpu.to_device(Device::OpenCl);
        let b_cl = b_cpu.to_device(Device::OpenCl);

        let cl_time = measure_opencl(args.warmup, args.iterations, || {
            let _c = a_cl.matmul_transposed(&b_cl);
        });

        print_result(
            format!("MatMul^T [{},{},{}]", m, k, n).as_str(),
            format!("M={} K={} N={}", m, k, n).as_str(),
            cpu_time,
            cl_time,
        );
    }
}

fn bench_matmul_slice(m: usize, rows: usize, cols: usize, args: &Args) {
    let shape_a = Shape::new(vec![m, cols]);
    let a_cpu = Tensor::new(vec![0.5; m * cols], shape_a.clone());
    let slice_data = vec![0.5; rows * cols];

    // CPU
    let cpu_time = measure(args.warmup, args.iterations, || {
        let _c = a_cpu.matmul_slice(&slice_data, rows, cols);
    });

    #[cfg(feature = "opencl")]
    {
        let a_cl = a_cpu.to_device(Device::OpenCl);
        // Note: matmul_slice takes a slice &[f32] which is on CPU.
        // The backend handles the upload if needed.
        let cl_time = measure_opencl(args.warmup, args.iterations, || {
            let _c = a_cl.matmul_slice(&slice_data, rows, cols);
        });

        print_result(
            "MatMulSlice",
            format!("M={} R={} C={}", m, rows, cols).as_str(),
            cpu_time,
            cl_time,
        );
    }
}

fn bench_rms_norm(rows: usize, cols: usize, args: &Args) {
    let shape = Shape::new(vec![rows, cols]);
    let x_cpu = Tensor::new(vec![0.5; rows * cols], shape.clone());
    let w_cpu = Tensor::new(vec![0.5; cols], Shape::new(vec![cols]));

    let cpu_time = measure(args.warmup, args.iterations, || {
        let _o = x_cpu.rms_norm(&w_cpu, 1e-5);
    });

    #[cfg(feature = "opencl")]
    {
        let x_cl = x_cpu.to_device(Device::OpenCl);
        let w_cl = w_cpu.to_device(Device::OpenCl);

        let cl_time = measure_opencl(args.warmup, args.iterations, || {
            let _o = x_cl.rms_norm(&w_cl, 1e-5);
        });

        print_result(
            "RMS Norm",
            format!("{}x{}", rows, cols).as_str(),
            cpu_time,
            cl_time,
        );
    }
}

fn bench_rope(batch: usize, heads: usize, seq_len: usize, args: &Args) {
    let head_dim = 128; // Fixed head dim
    let shape = Shape::new(vec![batch, seq_len, heads, head_dim]); // Check logic, typically [Batch, Seq, Head, Dim] or [Batch, Head, Seq, Dim]
    // Based on implementation: dims are flattened, last is head_dim.
    // Let's assume standard Llama: [Batch, Seq_Len, N_Heads, Head_Dim]

    let num_el = batch * seq_len * heads * head_dim;
    let mut x_cpu = Tensor::new(vec![0.5; num_el], shape.clone());

    let cpu_time = measure(args.warmup, args.iterations, || {
        let mut t = x_cpu.clone(); // Clone to measure inplace op cost + op cost
        t.apply_rope_inplace(0);
    });

    #[cfg(feature = "opencl")]
    {
        let x_cl = x_cpu.to_device(Device::OpenCl);
        let cl_time = measure_opencl(args.warmup, args.iterations, || {
            // Need to clone to avoid mutating the source for next iteration
            // But cloning on GPU is also an op.
            // Better: just run inplace and ignore data corruption for benchmark?
            // Or re-upload? Re-upload is slow.
            // Copy kernel is fast.
            // Let's rely on copy_from
            let mut t = llmrs::backend::opencl::OpenClBackend::new().copy_from(&x_cl);
            t.apply_rope_inplace(0);
        });
        print_result(
            "RoPE",
            format!("B={} S={} H={}", batch, seq_len, heads).as_str(),
            cpu_time,
            cl_time,
        );
    }
}

fn bench_softmax(rows: usize, cols: usize, args: &Args) {
    let shape = Shape::new(vec![rows, cols]);
    let x_cpu = Tensor::new(vec![0.5; rows * cols], shape.clone());

    let cpu_time = measure(args.warmup, args.iterations, || {
        let _o = x_cpu.softmax();
    });

    #[cfg(feature = "opencl")]
    {
        let x_cl = x_cpu.to_device(Device::OpenCl);
        let cl_time = measure_opencl(args.warmup, args.iterations, || {
            let _o = x_cl.softmax();
        });
        print_result(
            "Softmax",
            format!("{}x{}", rows, cols).as_str(),
            cpu_time,
            cl_time,
        );
    }
}

fn measure<F>(warmup: usize, iter: usize, mut f: F) -> f64
where
    F: FnMut(),
{
    for _ in 0..warmup {
        f();
    }
    let start = Instant::now();
    for _ in 0..iter {
        f();
    }
    let dur = start.elapsed();
    dur.as_secs_f64() * 1000.0 / iter as f64
}

#[cfg(feature = "opencl")]
fn measure_opencl<F>(warmup: usize, iter: usize, mut f: F) -> f64
where
    F: FnMut(),
{
    let queue = llmrs::backend::opencl::cl_context().queue();
    for _ in 0..warmup {
        f();
    }
    queue.finish().unwrap(); // Ensure warmup done

    let start = Instant::now();
    for _ in 0..iter {
        f();
        // Force sync for accurate timing if the kernel doesn't automatically.
        // Our backend usually syncs at end of ops (queue.finish) but let's be sure.
        queue.finish().unwrap();
    }
    let dur = start.elapsed();
    dur.as_secs_f64() * 1000.0 / iter as f64
}

fn print_result(op: &str, size: &str, cpu_ms: f64, cl_ms: f64) {
    let speedup = cpu_ms / cl_ms;
    println!(
        "{:<25} | {:<20} | {:<15.3} | {:<15.3} | {:<10.2}x",
        op, size, cpu_ms, cl_ms, speedup
    );
}
