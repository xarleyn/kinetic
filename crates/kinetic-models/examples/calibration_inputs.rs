//! Capture real matrix inputs through the unmodified model forward path.
//! cargo run --release -p kinetic-models --features calibration --example calibration_inputs -- model.gguf corpus.txt NEW_DIR [chunks=8] [context=512] [--dry-run]

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use kinetic_cpu::calibration::{capture, Target};
use kinetic_gguf::GgufFile;
use kinetic_quant::Matrix;
use kinetic_tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err("usage: calibration_inputs model.gguf corpus.txt NEW_DIR [chunks=8] [context=512] [--dry-run]".into());
    }
    let dry = args.iter().any(|s| s == "--dry-run");
    let positional: Vec<_> = args.iter().filter(|s| s.as_str() != "--dry-run").collect();
    if !(3..=5).contains(&positional.len()) {
        return Err("expected model.gguf corpus.txt NEW_DIR [chunks] [context]".into());
    }
    let chunks: usize = positional.get(3).map_or(Ok(8), |s| s.parse())?;
    let context: usize = positional.get(4).map_or(Ok(512), |s| s.parse())?;
    if chunks == 0 || context < 2 || chunks.checked_mul(context).is_none_or(|n| n > 65536) {
        return Err("capture requires 1..65536 total tokens and context >= 2".into());
    }
    let output = PathBuf::from(positional[2]);
    if output.exists() {
        return Err("output directory already exists".into());
    }
    let file = GgufFile::open(positional[0])?;
    let tokenizer = Tokenizer::from_gguf(file.header())?;
    let text = fs::read_to_string(positional[1])?;
    let tokens = tokenizer.encode(&text, false);
    if tokens.len() < chunks * context {
        return Err("corpus is too short".into());
    }
    let names: Vec<_> = [0, 15, 29]
        .into_iter()
        .flat_map(|i| {
            [
                format!("blk.{i}.attn_q.weight"),
                format!("blk.{i}.ffn_down.weight"),
            ]
        })
        .collect();
    let targets = names
        .iter()
        .map(|name| {
            let t = file
                .header()
                .tensor(name)
                .ok_or("missing selected tensor")?;
            if t.dims.len() != 2 {
                return Err("selected tensor must be a matrix");
            }
            Ok(Target::new(&Matrix::new(
                t.ty,
                file.tensor_data(t),
                t.dims[1] as usize,
                t.dims[0] as usize,
            )))
        })
        .collect::<Result<Vec<_>, &str>>()?;
    if dry {
        println!(
            "DRY RUN: capture {} tokens for {} matrices into {}; source files are read only",
            chunks * context,
            names.len(),
            output.display()
        );
        return Ok(());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(6)
        .build_global()?;
    let model = kinetic_models::load(&file)?;
    let bos = tokenizer.encode("", false);
    let batches: Vec<_> = tokens[..chunks * context]
        .chunks(context)
        .map(|ids| {
            let mut ids = ids.to_vec();
            if let [bos] = bos[..] {
                ids[0] = bos;
            }
            ids
        })
        .collect();
    let run = || {
        batches
            .iter()
            .map(|ids| model.run(&mut model.new_state(), ids, false).logits)
            .collect::<Vec<_>>()
    };
    let expected = run();
    let (actual, inputs) = capture(targets, chunks * context, run);
    if actual.len() != expected.len()
        || actual.iter().zip(&expected).any(|(a, b)| {
            a.len() != b.len() || a.iter().zip(b).any(|(a, b)| a.to_bits() != b.to_bits())
        })
    {
        return Err("capture changed model logits".into());
    }
    if inputs.iter().any(|x| {
        x.values.len() != chunks * context * x.cols
            || x.seen_rows != chunks * context
            || x.calls != chunks
    }) {
        return Err("missing, duplicated, or truncated selected matrix inputs".into());
    }
    fs::create_dir_all(&output)?;
    for (name, input) in names.iter().zip(inputs) {
        let mut out = BufWriter::new(File::create(output.join(format!("{name}.act")))?);
        out.write_all(b"KACT0001")?;
        out.write_all(&(input.seen_rows as u32).to_le_bytes())?;
        out.write_all(&(input.cols as u32).to_le_bytes())?;
        for value in input.values {
            out.write_all(&value.to_le_bytes())?;
        }
        out.flush()?;
    }
    let mut ids = BufWriter::new(File::create(output.join("tokens.u32le"))?);
    for batch in &batches {
        for token in batch {
            ids.write_all(&token.to_le_bytes())?;
        }
    }
    ids.flush()?;
    let checked: usize = actual.iter().map(Vec::len).sum();
    fs::write(output.join("capture.txt"), format!("model={}\ncorpus={}\nchunks={chunks}\ncontext={context}\ninput_rows_per_matrix={}\ncompared_logits={checked}\nmax_logit_difference=0\n", positional[0], positional[1], chunks * context))?;
    println!("Captured {} actual input rows per matrix; {checked} logits bit-identical with capture disabled/enabled", chunks * context);
    Ok(())
}
