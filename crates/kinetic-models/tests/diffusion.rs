//! Structural checks on the scaled FP32 LFM2 MoE exported by
//! scripts/diffusion_reference.py. Set KINETIC_TEST_DIFFUSION_GGUF and run
//! cargo test --release -p kinetic-models --test diffusion -- --ignored.
use kinetic_gguf::GgufFile;
use kinetic_models::diffusion::{Config, Diffusion, Step};

fn open() -> GgufFile {
    GgufFile::open(std::env::var("KINETIC_TEST_DIFFUSION_GGUF").expect("set fixture path")).unwrap()
}

#[test]
#[ignore = "needs KINETIC_TEST_DIFFUSION_GGUF"]
fn prefix_is_causal_canvas_sees_future_and_passes_do_not_share_state() {
    let file = open();
    let model = Diffusion::load(&file, vec![0.125; 32]).unwrap();
    let prefix = [256, 11, 23, 47];
    let a = model.forward(&prefix, &[13, 29, 53], &[false; 3]).unwrap();
    let b = model.forward(&prefix, &[13, 29, 109], &[false; 3]).unwrap();
    let d = model.n_embd();
    assert_eq!(&a.hidden[..prefix.len() * d], &b.hidden[..prefix.len() * d]);
    let difference = a.logits[..model.n_vocab()]
        .iter()
        .zip(&b.logits)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        difference > 1e-4,
        "earliest canvas row must see the changed future row"
    );
    let changed_prefix = model
        .forward(&[256, 11, 23, 99], &[13, 29, 53], &[false; 3])
        .unwrap();
    assert_eq!(&a.hidden[..3 * d], &changed_prefix.hidden[..3 * d]);
    let again = model.forward(&prefix, &[13, 29, 53], &[false; 3]).unwrap();
    assert_eq!(a.hidden, again.hidden);
    assert_eq!(a.logits, again.logits);
}

#[test]
#[ignore = "needs KINETIC_TEST_DIFFUSION_GGUF"]
fn masks_replace_token_embeddings_and_invalid_inputs_fail() {
    let file = open();
    let model = Diffusion::load(&file, vec![0.125; 32]).unwrap();
    let a = model
        .forward(&[256, 17], &[13, 29, 53], &[true, false, true])
        .unwrap();
    let b = model
        .forward(&[256, 17], &[79, 29, 101], &[true, false, true])
        .unwrap();
    assert_eq!(a.logits, b.logits);
    assert_eq!(a.hidden, b.hidden);
    assert!(model.forward(&[], &[1], &[true]).is_err());
    assert!(model.forward(&[1], &[1, 2], &[true]).is_err());
    assert!(model
        .forward(&[1], &[model.n_vocab() as u32], &[true])
        .is_err());
    assert!(Diffusion::load(&file, vec![0.; 31]).is_err());
    assert!(Diffusion::load(&file, vec![f32::NAN; 32]).is_err());
    assert!(kinetic_models::load(&file)
        .err()
        .unwrap()
        .to_string()
        .contains("use kinetic diffuse"));
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
#[ignore = "needs KINETIC_TEST_DIFFUSION_GGUF"]
fn reused_prefix_logits_are_bit_identical_across_boundaries_and_masks() {
    let file = open();
    let model = Diffusion::load(&file, vec![0.125; 32]).unwrap();
    for length in [1, 2, 3, 9, 37, 63, 64, 65, 257] {
        let prefix: Vec<u32> = (0..length).map(|i| 10 + (i * 37 % 240) as u32).collect();
        let cached = model.prepare_prefix(&prefix).unwrap();
        for count in [1, 16, 32, 64] {
            let canvas: Vec<u32> = (0..count).map(|i| 13 + (i * 53 % 240) as u32).collect();
            for style in 0..3 {
                let masked: Vec<bool> = (0..count)
                    .map(|i| style == 0 || (style == 1 && i % 2 == 0))
                    .collect();
                let full = model.forward(&prefix, &canvas, &masked).unwrap().logits;
                let reused = cached.forward(&canvas, &masked).unwrap();
                assert_eq!(
                    bits(&full),
                    bits(&reused),
                    "prefix={length}, canvas={count}, mask={style}"
                );
            }
        }
        // A -> B -> A must leave the immutable conv boundary unchanged.
        let a = cached.forward(&[17, 29, 53], &[false; 3]).unwrap();
        cached.forward(&[79, 113], &[true, false]).unwrap();
        assert_eq!(
            bits(&a),
            bits(&cached.forward(&[17, 29, 53], &[false; 3]).unwrap())
        );
        assert!(cached.forward(&[1], &[]).is_err());
        assert!(cached.forward(&[model.n_vocab() as u32], &[false]).is_err());
    }
    assert!(model.prepare_prefix(&[]).is_err());
    assert!(model.prepare_prefix(&[model.n_vocab() as u32]).is_err());
}

#[derive(Debug, PartialEq)]
struct Trace {
    block: usize,
    step: usize,
    prefix: Vec<u32>,
    before: Vec<u32>,
    masked: Vec<bool>,
    logits: Vec<u32>,
    candidates: Vec<u32>,
    confidence: Vec<u32>,
    committed: Vec<usize>,
    after: Vec<u32>,
}

fn capture(s: &Step<'_>) -> Trace {
    Trace {
        block: s.block,
        step: s.step,
        prefix: s.prefix.to_vec(),
        before: s.canvas_before.to_vec(),
        masked: s.masked_before.to_vec(),
        logits: bits(s.logits),
        candidates: s.candidates.to_vec(),
        confidence: bits(s.log_confidence),
        committed: s.committed.to_vec(),
        after: s.canvas_after.to_vec(),
    }
}

#[test]
#[ignore = "needs KINETIC_TEST_DIFFUSION_GGUF"]
fn reuse_preserves_every_sampler_decision_and_rebuilds_each_block() {
    let file = open();
    let model = Diffusion::load(&file, vec![0.125; 32]).unwrap();
    for (canvas, steps, maximum, length) in [
        (1, 16, 3, 9),
        (16, 1, 39, 37),
        (16, 4, 39, 257),
        (32, 4, 71, 37),
        (64, 4, 135, 257),
    ] {
        let prompt: Vec<u32> = (0..length).map(|i| 10 + (i * 37 % 240) as u32).collect();
        let config = Config {
            canvas,
            steps,
            max_new_tokens: maximum,
            prefix_window: 8,
            pad: 258,
            eos: vec![257],
        };
        let mut full = Vec::new();
        let mut reused = Vec::new();
        let a = model
            .generate(&prompt, &config, |s| {
                full.push(capture(s));
                Ok(())
            })
            .unwrap();
        let b = model
            .generate_reusing_prefix(&prompt, &config, |s| {
                reused.push(capture(s));
                Ok(())
            })
            .unwrap();
        assert_eq!(
            full, reused,
            "canvas={canvas}, steps={steps}, prefix={length}"
        );
        assert_eq!(a.tokens, b.tokens);
        assert_eq!(a.forward_calls, b.forward_calls);
        assert_eq!(a.prefix_builds, 0);
        assert_eq!(b.prefix_builds, full.last().unwrap().block + 1);
    }
}
