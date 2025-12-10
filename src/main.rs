mod backend;
mod core;

use anyhow::Result;
use clap::Parser;
use log::*;
use log::{error, info};
use std::io::Write; // flush를 위해 필요
use std::path::PathBuf;
use tokenizers::Tokenizer; // [New] log 매크로 사용

use crate::core::config::LlamaConfig;
use crate::core::loader::Loader;
use crate::core::shape::Shape;
use crate::core::tensor::Tensor;

/// Simple Llama 3.2 Inference Engine in Rust
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

    #[arg(short = 'n', long, default_value_t = 99)]
    steps: usize,
}

fn main() -> Result<()> {
    // 1. 로거 초기화 (Custom Format)
    // 환경변수 RUST_LOG가 없으면 "info" 레벨로 설정
    if std::env::var("RUST_LOG").is_err() {
        unsafe { std::env::set_var("RUST_LOG", "info") };
    }

    // 로그 포맷을 깔끔하게 커스텀 (시간, 레벨 색상 등)
    env_logger::Builder::from_default_env()
        .format(|buf, record| {
            use std::io::Write;
            let ts = buf.timestamp_seconds();
            let level_style = buf.default_level_style(record.level());
            writeln!(
                buf,
                "[{} {}] {}",
                ts,
                level_style.value(record.level()),
                record.args()
            )
        })
        .init();

    // 2. 인자 파싱
    let args = Args::parse();
    info!("Starting Rust LLM...");
    info!("Args: {:?}", args);

    // 3. Config 로드
    info!("Loading Config from {:?}...", args.config);
    let config_file = std::fs::File::open(&args.config)
        .map_err(|_| anyhow::anyhow!("Config file not found: {:?}", args.config))?;
    let config: LlamaConfig = serde_json::from_reader(std::io::BufReader::new(config_file))?;

    // 4. Model 로드
    info!("Loading Model from {:?}...", args.model);
    let loader = Loader::new(&args.model)?;
    let model = loader.load_model(&config)?;
    info!("Model loaded successfully.");

    // 5. Tokenizer 로드
    info!("Loading Tokenizer from {:?}...", args.tokenizer);
    let tokenizer = Tokenizer::from_file(&args.tokenizer).map_err(|e| anyhow::anyhow!(e))?;

    // 6. Encode Prompt
    info!("Encoding Prompt: \"{}\"", args.prompt);
    let encoding = tokenizer
        .encode(args.prompt.clone(), true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let mut input_ids: Vec<u32> = encoding.get_ids().to_vec();

    // 7. Generation Loop
    info!("Generating {} tokens...", args.steps);
    println!("\n--- Output ---");

    // 프롬프트 먼저 출력 (파란색 효과 등은 제거하고 순수 텍스트로)
    println!("{}", args.prompt);
    std::io::stdout().flush()?;

    let start_gen = std::time::Instant::now();

    // Llama 3의 종료 토큰 ID (보통 128001: <|end_of_text|> 혹은 128009: <|eot_id|>)
    // Tokenizer json을 봐야 정확하지만, Llama 3 기준 128001이나 128009입니다.
    let eos_token_ids = [128001u32, 128009u32];

    let mut cur_pos = 0;

    for _ in 0..args.steps {
        let seq_len = input_ids.len();

        let (input_chunk, chunk_len) = if cur_pos == 0 {
            (input_ids.clone(), seq_len)
        } else {
            (vec![*input_ids.last().unwrap()], 1)
        };

        // Embedding Lookup
        let hidden = config.hidden_size;
        let mut embed_data = Vec::with_capacity(chunk_len * hidden);
        let all_embeds = model.embed_tokens.data();

        for &id in &input_chunk {
            let start = (id as usize) * hidden;
            if start + hidden > all_embeds.len() {
                anyhow::bail!("Token ID {} out of vocab range", id);
            }
            embed_data.extend_from_slice(&all_embeds[start..start + hidden]);
        }
        let input_tensor = Tensor::new(embed_data, Shape::new(vec![chunk_len, hidden]));

        // Forward
        let logits = model.forward(&input_tensor, cur_pos);

        // Greedy Sampling
        let logits_data = logits.data();
        let vocab = config.vocab_size;
        let last_logits = &logits_data[(chunk_len - 1) * vocab..];

        let (next_token, _) = last_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        trace!("Next token ID: {}", next_token);

        if eos_token_ids.contains(&(next_token as u32)) {
            info!("EOS token detected. Stopping.");
            break;
        }

        // Update & Print
        input_ids.push(next_token as u32);
        cur_pos += chunk_len;

        // 토큰 출력은 로그가 아니라 실제 출력이므로 print! 사용
        let word = tokenizer
            .decode(&[next_token as u32], true)
            .unwrap_or_else(|_| "".to_string());
        trace!("Decoded token: {}", word);

        print!("{}", word);
        std::io::stdout().flush()?;
    }

    let elapsed = start_gen.elapsed();
    println!("\n--------------"); // 줄바꿈

    // 결과 통계는 다시 로그로
    info!(
        "Generation finished in {:.2}s ({:.2} tok/s)",
        elapsed.as_secs_f64(),
        args.steps as f64 / elapsed.as_secs_f64()
    );

    Ok(())
}
