#![cfg_attr(target_arch = "aarch64", feature(stdarch_neon_dotprod))]

mod backend;
mod core;
mod profile;

use anyhow::Result;
use clap::Parser;
use log::*;
use log::{error, info};
use rand::Rng;
use std::collections::HashSet;
use std::io::Write; // flush를 위해 필요
use std::path::PathBuf;
use tokenizers::Tokenizer; // [New] 랜덤 샘플링을 위해 필요

use crate::backend::Device;
use crate::core::config::LlamaConfig;
use crate::core::loader::Loader;
use crate::core::shape::Shape;
use crate::core::tensor::Tensor;
use crate::profile::Profiler;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "models/model.safetensors")]
    model: PathBuf,

    #[arg(short, long, default_value = "models/config.json")]
    config: PathBuf,

    #[arg(short, long, default_value = "models/tokenizer.json")]
    tokenizer: PathBuf,

    #[arg(short, long, default_value = "Hello world")]
    prompt: String,

    #[arg(short = 'n', long, default_value_t = 200)]
    steps: usize,

    // [옵션] 반복 페널티 (기본 1.1 -> 1.2로 상향 추천)
    #[arg(short = 'r', long, default_value_t = 1.2)]
    repeat_penalty: f32,

    #[arg(short = 'w', long, default_value_t = 64)]
    penalty_window: usize,

    // Temperature (창의성 조절: 높을수록 다양함, 낮을수록 보수적)
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    // Top-P (Nucleus Sampling: 상위 P% 확률 내에서만 뽑기)
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,

    // 디바이스 선택 옵션 (cpu, gpu)
    #[arg(long, default_value = "cpu")]
    device: String,
}

fn main() -> Result<()> {
    if std::env::var("RUST_LOG").is_err() {
        unsafe { std::env::set_var("RUST_LOG", "info") };
    }
    env_logger::Builder::from_default_env()
        .format(|buf, record| {
            use std::io::Write;
            writeln!(buf, "[{}] {}", record.level(), record.args())
        })
        .init();

    let args = Args::parse();

    let config_file = std::fs::File::open(&args.config)?;
    let config: LlamaConfig = serde_json::from_reader(std::io::BufReader::new(config_file))?;

    debug!("Init loader");
    let loader = Loader::new(&args.model)?;

    debug!("Load model");
    let mut model = loader.load_model(&config)?;
    let device = match args.device.to_lowercase().as_str() {
        "gpu" | "opencl" => Device::OpenCl,
        _ => Device::Cpu,
    };

    if device == Device::OpenCl {
        // Phase 2에서 만든 OpenCL 초기화가 내부적으로 수행됨
        model = model.to_device(Device::OpenCl);
    }

    debug!("Init tokenizer");
    let tokenizer = Tokenizer::from_file(&args.tokenizer).map_err(|e| anyhow::anyhow!(e))?;
    profile!("0. Model Loading"); // 블록이 끝나면 자동 기록됨

    #[cfg(feature = "opencl")]
    {
        println!("Testing OpenCL Shared Memory...");
        let cl_backend = crate::backend::opencl::OpenClBackend::new();
        let shape = Shape::new(vec![10]);

        // 1. Shared Tensor 할당
        let mut tensor = cl_backend.allocate_shared(&shape);
        println!("Tensor created on: {:?}", tensor.device());

        // 2. CPU에서 데이터 쓰기 (직접 포인터 접근)
        {
            let data = tensor.data_mut();
            for i in 0..10 {
                data[i] = 10.0; // 초기값 10.0
            }
            println!("CPU wrote 10.0 to all elements.");
        }

        // 3. GPU 커널 실행 (각 원소에 +1.0)
        // 데이터 복사(WriteBuffer/ReadBuffer) 없이 커널만 실행합니다.
        println!("Launching GPU kernel (add +1.0)...");
        cl_backend.launch_dummy_kernel(&tensor);

        // 4. CPU에서 데이터 확인
        {
            let data = tensor.data();
            println!("Result check: {:?}", data);
            if data[0] == 11.0 {
                println!(">> SUCCESS: Zero-Copy Shared Memory works! (10.0 -> 11.0)");
            } else {
                println!(">> FAILURE: Value mismatch. Got {}", data[0]);
            }
        }

        println!("Testing OpenCL MatMul (Phase 2)...");
        let cl_backend = crate::backend::opencl::OpenClBackend::new();

        // A: [2, 4]
        let shape_a = Shape::new(vec![2, 4]);
        let mut a = cl_backend.allocate_shared(&shape_a);
        {
            let d = a.data_mut();
            // Row 1: 1, 1, 1, 1
            // Row 2: 2, 2, 2, 2
            for i in 0..4 {
                d[i] = 1.0;
            }
            for i in 4..8 {
                d[i] = 2.0;
            }
        }

        // B: [4, 2] (Normal)
        let shape_b = Shape::new(vec![4, 2]);
        let mut b = cl_backend.allocate_shared(&shape_b);
        {
            let d = b.data_mut();
            // Col 1: 1, 1, 1, 1
            // Col 2: 2, 2, 2, 2 (interleaved)
            // [1, 2, 1, 2, 1, 2, 1, 2]
            for i in 0..8 {
                d[i] = if i % 2 == 0 { 1.0 } else { 2.0 };
            }
        }

        // Execute GPU MatMul
        // A @ B -> [2, 2]
        // [1,1,1,1] . [1,1,1,1] = 4
        // [1,1,1,1] . [2,2,2,2] = 8
        // [2,2,2,2] . [1,1,1,1] = 8
        // [2,2,2,2] . [2,2,2,2] = 16
        let c = a.matmul(&b);

        println!("MatMul Result: {:?}", c.data());

        let res = c.data();
        if res[0] == 4.0 && res[1] == 8.0 && res[2] == 8.0 && res[3] == 16.0 {
            println!(">> SUCCESS: GPU MatMul works correctly!");
        } else {
            println!(">> FAILURE: MatMul results incorrect.");
        }

        println!("Testing OpenCL Ops (Phase 3: RMSNorm & SiLU)...");

        // 1. RMS Norm Test
        // Input: [1.0, 2.0, 3.0, 4.0]
        // Weight: [1.0, 1.0, 1.0, 1.0]
        // Squares: 1+4+9+16 = 30. Mean = 7.5. RMS = sqrt(7.5) ≈ 2.7386
        // Expected: [1/2.738, 2/2.738, ...] ≈ [0.365, 0.730, 1.095, 1.460]
        let shape_vec = Shape::new(vec![4]);
        let mut vec_x = cl_backend.allocate_shared(&shape_vec);
        let mut vec_w = cl_backend.allocate_shared(&shape_vec);
        {
            let d = vec_x.data_mut();
            d[0] = 1.0;
            d[1] = 2.0;
            d[2] = 3.0;
            d[3] = 4.0;
            let w = vec_w.data_mut();
            for i in 0..4 {
                w[i] = 1.0;
            }
        }

        let norm_out = vec_x.rms_norm(&vec_w, 1e-5);
        let res_norm = norm_out.data();
        println!("RMSNorm Result: {:?}", res_norm);
        if (res_norm[0] - 0.365).abs() < 0.01 {
            println!(">> SUCCESS: RMSNorm working.");
        } else {
            println!(">> FAILURE: RMSNorm mismatch.");
        }

        // 2. SiLU Test
        // Gate: [2.0], Up: [10.0]
        // SiLU(2.0) = 2.0 / (1 + exp(-2)) ≈ 2.0 / (1 + 0.135) ≈ 2.0 / 1.135 ≈ 1.761
        // Result = 1.761 * 10.0 = 17.61
        let shape_sc = Shape::new(vec![1]);
        let mut gate = cl_backend.allocate_shared(&shape_sc);
        let mut up = cl_backend.allocate_shared(&shape_sc);
        {
            gate.data_mut()[0] = 2.0;
            up.data_mut()[0] = 10.0;
        }
        gate.silu_mul_inplace(&up);
        let res_silu = gate.data()[0];
        println!("SiLU Result: {:.4}", res_silu);

        if (res_silu - 17.615).abs() < 0.1 {
            println!(">> SUCCESS: SiLU working.");
        } else {
            println!(">> FAILURE: SiLU mismatch.");
        }
    }

    let encoding = tokenizer
        .encode(args.prompt.clone(), true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let mut input_ids: Vec<u32> = encoding.get_ids().to_vec();

    println!("\n--- Output ---");
    print!("{}", args.prompt);
    std::io::stdout().flush()?;

    let start_gen = std::time::Instant::now();
    let eos_token_ids = [128001u32, 128009u32]; // Llama 3 EOS
    let mut cur_pos = 0;
    let mut rng = rand::thread_rng(); // 랜덤 생성기

    {
        profile!("1. Total Inference");

        for _ in 0..args.steps {
            let input_chunk: Vec<u32>;
            let chunk_len: usize;
            {
                profile!("2. Step Generation");
                let seq_len = input_ids.len();
                (input_chunk, chunk_len) = if cur_pos == 0 {
                    (input_ids.clone(), seq_len)
                } else {
                    (vec![*input_ids.last().unwrap()], 1)
                };
            }

            let input_tensor: Tensor;
            {
                profile!("3. Input Preparation");
                // Embedding Look-up은 CPU에서 수행하는 것이 효율적일 수 있으나,
                // 모델 구조상 embed_tokens가 GPU에 가 있으면 Look-up을 위해
                // 인덱스를 GPU로 보내거나, embed_tokens를 CPU에 남겨둬야 합니다.
                // 현재 구조: embed_tokens도 GPU로 이동함.

                // [간단한 해결책]
                // Embeddings는 보통 CPU에서 룩업해서 GPU로 올리는게 일반적입니다.
                // 하지만 여기서는 model.embed_tokens도 to_device로 GPU로 가버렸습니다.
                // 따라서 GPU 텐서에서 슬라이싱을 해야 하는데, 현재 백엔드엔 Gather 커널이 없습니다.

                // [수정 제안]
                // Loader나 Model 구조상 Embeddings는 CPU에 남겨두거나,
                // 여기서는 임시로 CPU에서 룩업 후 GPU로 올리는 방식을 씁니다.
                // (주의: model.embed_tokens가 GPU면 data() 접근이 Shared Memory라 가능하긴 함)

                let hidden = config.hidden_size;
                let mut embed_data = Vec::with_capacity(chunk_len * hidden);

                // Zero-Copy 덕분에 GPU에 있어도 data()로 읽기 가능!
                let all_embeds = model.embed_tokens.data();
                for &id in &input_chunk {
                    let start = (id as usize) * hidden;
                    embed_data.extend_from_slice(&all_embeds[start..start + hidden]);
                }

                // [핵심] 입력 텐서를 타겟 디바이스로 생성/이동
                input_tensor =
                    Tensor::new(embed_data, Shape::new(vec![chunk_len, hidden])).to_device(device);
            }

            // Forward
            let logits: Tensor;
            {
                profile!("4. Forward Pass");
                // input_tensor가 device에 있으므로 내부 연산도 device에서 수행됨
                logits = model.forward(&input_tensor, cur_pos);
            }

            let mut next_token_logits: Vec<f32>;
            {
                profile!("5. Logits Extraction");
                let logits_data = logits.data();
                let vocab = config.vocab_size;

                // 마지막 토큰의 Logits 추출
                let start_idx = (chunk_len - 1) * vocab;
                next_token_logits = logits_data[start_idx..start_idx + vocab].to_vec();

                // 1. Repetition Penalty 적용
                let penalty = args.repeat_penalty;
                if penalty > 1.0 {
                    let window_start =
                        if args.penalty_window > 0 && input_ids.len() > args.penalty_window {
                            input_ids.len() - args.penalty_window
                        } else {
                            0
                        };
                    let tokens_to_penalize: HashSet<_> = input_ids[window_start..].iter().collect();

                    for &id in tokens_to_penalize {
                        let id = id as usize;
                        if id < next_token_logits.len() {
                            let score = next_token_logits[id];
                            if score < 0.0 {
                                next_token_logits[id] = score * penalty;
                            } else {
                                next_token_logits[id] = score / penalty;
                            }
                        }
                    }
                }

                // 2. Temperature 적용 (Logits 스케일링)
                let temp = args.temperature;
                if (temp - 1.0).abs() > 1e-6 {
                    for logit in next_token_logits.iter_mut() {
                        *logit /= temp;
                    }
                }
            }

            // 3. Softmax (확률 변환)
            {
                profile!("6. Sampling");
                let max_logit = next_token_logits.iter().fold(f32::MIN, |a, &b| a.max(b));
                let mut probs: Vec<f32> = next_token_logits
                    .iter()
                    .map(|l| (l - max_logit).exp())
                    .collect();
                let sum_probs: f32 = probs.iter().sum();
                for p in probs.iter_mut() {
                    *p /= sum_probs;
                }

                // 4. Top-P Sampling (Nucleus)
                let next_token = if args.top_p < 1.0 {
                    // [최적화] 전체 Vocab(128k)을 정렬하면 느림.
                    // 확률이 너무 낮은(예: 0.0) 토큰은 Top-P에 들어갈 가망이 없으므로 제외.
                    let threshold = 1e-5; // 0.00001보다 작은 확률은 무시

                    let mut sorted_probs: Vec<(f32, usize)> = probs
                        .iter()
                        .enumerate()
                        .filter(|(_, p)| p > &&threshold) // [추가됨] 가지치기
                        .map(|(i, &p)| (p, i))
                        .collect();

                    // 확률 높은 순으로 정렬 (이제 개수가 몇 개 안 되어 매우 빠름)
                    sorted_probs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

                    // 모든 확률이 threshold보다 낮아 후보가 없다면, 전체를 다시 포함시킵니다.
                    if sorted_probs.is_empty() {
                        sorted_probs = probs.iter().enumerate().map(|(i, &p)| (p, i)).collect();
                    }

                    let mut cumulative_prob = 0.0;

                    // sorted_probs가 비어있지 않음을 보장했으므로 안전합니다.
                    let mut cutoff_index = sorted_probs.len().saturating_sub(1);
                    for (i, (prob, _)) in sorted_probs.iter().enumerate() {
                        cumulative_prob += prob;
                        if cumulative_prob > args.top_p {
                            cutoff_index = i;
                            break;
                        }
                    }

                    // Top-P에 포함된 후보들 중에서 다시 확률 분포에 따라 랜덤 선택
                    let valid_candidates = &sorted_probs[0..=cutoff_index];
                    let total_valid_prob: f32 = valid_candidates.iter().map(|(p, _)| p).sum();

                    let mut r = rng.random::<f32>() * total_valid_prob;
                    let mut selected_token = valid_candidates[0].1;

                    for (prob, idx) in valid_candidates {
                        r -= prob;
                        if r <= 0.0 {
                            selected_token = *idx;
                            break;
                        }
                    }
                    selected_token
                } else {
                    // Top-P 미사용 시 단순 확률 비례 샘플링 (Roulette Wheel)
                    let mut r = rng.random::<f32>(); // 0.0 ~ 1.0
                    let mut selected_token = 0;
                    for (i, p) in probs.iter().enumerate() {
                        r -= p;
                        if r <= 0.0 {
                            selected_token = i;
                            break;
                        }
                    }
                    selected_token
                };

                // EOS Check
                if eos_token_ids.contains(&(next_token as u32)) {
                    break;
                }

                input_ids.push(next_token as u32);
                cur_pos += chunk_len;

                let word = tokenizer
                    .decode(&[next_token as u32], true)
                    .unwrap_or_else(|_| "".to_string());

                print!("{}", word);
                std::io::stdout().flush()?;
            }
        }
    }

    Profiler::print_stats();

    let elapsed = start_gen.elapsed();
    println!("\n--------------");
    info!(
        "Finished: {:.2} tok/s",
        args.steps as f64 / elapsed.as_secs_f64()
    );

    Ok(())
}
