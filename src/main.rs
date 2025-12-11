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

    // [New] Temperature (창의성 조절: 높을수록 다양함, 낮을수록 보수적)
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    // [New] Top-P (Nucleus Sampling: 상위 P% 확률 내에서만 뽑기)
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,
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
    // ... (로딩 로그 생략) ...

    let config_file = std::fs::File::open(&args.config)?;
    let config: LlamaConfig = serde_json::from_reader(std::io::BufReader::new(config_file))?;
    let loader = Loader::new(&args.model)?;
    let model = loader.load_model(&config)?;
    let tokenizer = Tokenizer::from_file(&args.tokenizer).map_err(|e| anyhow::anyhow!(e))?;
    profile!("0. Model Loading"); // 블록이 끝나면 자동 기록됨

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
                // Embedding
                let hidden = config.hidden_size;
                let mut embed_data = Vec::with_capacity(chunk_len * hidden);
                let all_embeds = model.embed_tokens.data();
                for &id in &input_chunk {
                    let start = (id as usize) * hidden;
                    embed_data.extend_from_slice(&all_embeds[start..start + hidden]);
                }
                input_tensor = Tensor::new(embed_data, Shape::new(vec![chunk_len, hidden]));
            }

            // Forward
            let logits: Tensor;
            {
                profile!("4. Forward Pass");
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

                    let mut cumulative_prob = 0.0;
                    let mut cutoff_index = sorted_probs.len() - 1;

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
