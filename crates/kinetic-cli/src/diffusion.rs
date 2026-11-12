//! Explicit CPU-only masked diffusion and a machine-readable comparison trace.
use clap::Args;
use kinetic_gguf::GgufFile;
use kinetic_models::diffusion::{Config, Diffusion, Step, MODE};
use kinetic_models::LoadError;
use kinetic_tokenizer::Tokenizer;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Args)]
pub struct DiffuseArgs {
    /// Marked lfm2moe GGUF; this command always runs on the CPU
    pub model: PathBuf,
    /// Safetensors containing the learned F32 mask_embedding vector
    #[arg(long)]
    mask_embedding: PathBuf,
    /// Raw UTF-8 prompt without a chat template
    #[arg(
        long,
        conflicts_with = "prompt_tokens",
        required_unless_present = "prompt_tokens"
    )]
    prompt: Option<String>,
    /// JSON array of prompt token IDs for reference comparisons
    #[arg(long, conflicts_with = "prompt", required_unless_present = "prompt")]
    prompt_tokens: Option<PathBuf>,
    /// Positions per canvas (1..64)
    #[arg(long, default_value_t = 16)]
    canvas: usize,
    /// Fixed denoising passes per block (1..16)
    #[arg(long, default_value_t = 4)]
    steps: usize,
    /// Most generated tokens (1..256)
    #[arg(short = 'n', long, default_value_t = 32)]
    max_new_tokens: usize,
    /// Generated history retained after the complete original prompt
    #[arg(long, default_value_t = 256)]
    prefix_window: usize,
    /// Reuse causal prefix KV and conv boundary states within each block
    #[arg(long)]
    prefix_reuse: bool,
    /// Timed generations after two warmups (1..100), excluding load and trace IO
    #[arg(long)]
    benchmark_runs: Option<usize>,
    /// New/empty directory for raw F32 logits and a JSON trace index
    #[arg(long)]
    trace_dir: Option<PathBuf>,
    /// Validate and print the plan without inference or trace writes
    #[arg(long)]
    dry_run: bool,
    #[arg(short = 't', long)]
    pub threads: Option<usize>,
}

pub fn run(args: DiffuseArgs) -> crate::Result<()> {
    if args.benchmark_runs.is_some_and(|n| !(1..=100).contains(&n)) {
        return Err("benchmark-runs must be in 1..100".into());
    }
    if let Some(dir) = &args.trace_dir {
        if dir.exists() && fs::read_dir(dir)?.next().is_some() {
            return Err("trace directory must be new or empty".into());
        }
    }
    let file = GgufFile::open(&args.model)?;
    let diffusion = Diffusion::load(&file, read_mask(&args.mask_embedding)?)?;
    let tokenizer = Tokenizer::from_gguf(file.header())?;
    let prompt: Vec<u32> = match args.prompt_tokens {
        Some(path) => serde_json::from_reader(File::open(path)?)?,
        None => tokenizer.encode(args.prompt.as_deref().unwrap_or_default(), true),
    };
    let pad = file
        .header()
        .get("tokenizer.ggml.padding_token_id")
        .and_then(|v| v.as_u64())
        .and_then(|v| u32::try_from(v).ok())
        .ok_or("diffusion requires a pad token in GGUF metadata")?;
    let config = Config {
        canvas: args.canvas,
        steps: args.steps,
        max_new_tokens: args.max_new_tokens,
        prefix_window: args.prefix_window,
        pad,
        eos: tokenizer.eos().into_iter().collect(),
    };
    config.validate(&prompt, diffusion.n_vocab())?;
    let settings = json!({"mode": MODE, "backend": "cpu", "sampler": "fixed-greedy-v1",
        "canvas": config.canvas, "steps": config.steps, "max_new_tokens": config.max_new_tokens,
        "prefix_window": config.prefix_window, "pad": pad, "eos": config.eos,
        "tie_break": "confidence descending, position ascending", "prefix_reuse": args.prefix_reuse,
        "vocab": diffusion.n_vocab(), "width": diffusion.n_embd()});
    if args.dry_run {
        println!(
            "{}",
            json!({"settings": settings, "prompt_tokens": prompt, "dry_run": true})
        );
        return Ok(());
    }
    if let Some(dir) = &args.trace_dir {
        fs::create_dir_all(dir)?;
    }
    let mut steps = Vec::new();
    let mut trace = |s: &Step<'_>| {
        let Some(dir) = &args.trace_dir else {
            return Ok(());
        };
        let name = format!("block-{:03}-step-{:02}.f32", s.block, s.step);
        let mut out = BufWriter::new(File::create_new(dir.join(&name)).map_err(io_error)?);
        for value in s.logits {
            out.write_all(&value.to_le_bytes()).map_err(io_error)?;
        }
        out.flush().map_err(io_error)?;
        steps.push(json!({"block": s.block, "step": s.step, "prefix": s.prefix,
            "canvas_before": s.canvas_before, "masked_before": s.masked_before,
            "logits_file": name, "logits_shape": [s.canvas_before.len(), diffusion.n_vocab()],
            "candidates": s.candidates,
            "log_confidence": s.log_confidence.iter().map(|v| v.is_finite().then_some(*v)).collect::<Vec<_>>(),
            "committed": s.committed, "canvas_after": s.canvas_after}));
        Ok(())
    };
    let result = if args.prefix_reuse {
        diffusion.generate_reusing_prefix(&prompt, &config, &mut trace)?
    } else {
        diffusion.generate(&prompt, &config, &mut trace)?
    };
    // Every sample includes a complete generation, including preparation of
    // every block's prefix. Model load, tokenization and trace writes are outside.
    let benchmark = if let Some(runs) = args.benchmark_runs {
        let infer = || {
            if args.prefix_reuse {
                diffusion.generate_reusing_prefix(&prompt, &config, |_| Ok(()))
            } else {
                diffusion.generate(&prompt, &config, |_| Ok(()))
            }
        };
        for _ in 0..2 {
            std::hint::black_box(infer()?);
        }
        let mut samples = Vec::with_capacity(runs);
        for _ in 0..runs {
            let start = Instant::now();
            let generated = infer()?;
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
            if generated.tokens != result.tokens || generated.forward_calls != result.forward_calls
            {
                return Err("diffusion benchmark generation changed its result".into());
            }
            std::hint::black_box(generated);
        }
        Some(
            json!({"warmups": 2, "samples_ms": samples, "includes_prefix_preparation": true,
            "excludes": ["model_load", "tokenization", "trace_io"]}),
        )
    } else {
        None
    };
    let output = json!({"settings": settings, "tokens": result.tokens,
        "text": tokenizer.decode(&result.tokens), "forward_calls": result.forward_calls,
        "prefix_builds": result.prefix_builds, "benchmark": benchmark});
    if let Some(dir) = &args.trace_dir {
        let index = json!({"schema_version": 1, "settings": settings, "prompt": prompt,
            "model_sha256": sha256(&args.model)?, "mask_sha256": sha256(&args.mask_embedding)?,
            "steps": steps, "output": output});
        serde_json::to_writer_pretty(File::create_new(dir.join("trace.json"))?, &index)?;
    }
    println!("{output}");
    Ok(())
}

fn io_error(e: std::io::Error) -> LoadError {
    LoadError(format!("diffusion trace: {e}"))
}

fn sha256(path: &Path) -> std::io::Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub(crate) fn read_mask(path: &Path) -> crate::Result<Vec<f32>> {
    if fs::metadata(path)?.len() > 1 << 20 {
        return Err("mask file exceeds 1 MiB".into());
    }
    let data = fs::read(path)?;
    let n = u64::from_le_bytes(data.get(..8).ok_or("truncated mask header")?.try_into()?);
    let end = usize::try_from(n)?
        .checked_add(8)
        .ok_or("mask header overflow")?;
    let header: Value = serde_json::from_slice(data.get(8..end).ok_or("truncated mask metadata")?)?;
    let t = &header["mask_embedding"];
    let shape = t["shape"]
        .as_array()
        .filter(|s| s.len() == 1)
        .ok_or("mask_embedding must be a vector")?;
    let width = shape[0].as_u64().ok_or("invalid mask width")?;
    if t["dtype"] != "F32" {
        return Err("mask_embedding must be F32".into());
    }
    let offsets = t["data_offsets"]
        .as_array()
        .filter(|s| s.len() == 2)
        .ok_or("invalid mask offsets")?;
    let start = usize::try_from(offsets[0].as_u64().ok_or("invalid mask offset")?)?
        .checked_add(end)
        .ok_or("mask offset overflow")?;
    let stop = usize::try_from(offsets[1].as_u64().ok_or("invalid mask offset")?)?
        .checked_add(end)
        .ok_or("mask offset overflow")?;
    let bytes = data.get(start..stop).ok_or("mask data outside file")?;
    if u64::try_from(bytes.len())? != width.checked_mul(4).ok_or("mask shape overflow")? {
        return Err("mask shape and data size disagree".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
