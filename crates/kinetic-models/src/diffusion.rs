//! Experimental CPU masked diffusion for LFM2 MoE. Each denoising pass
//! recomputes the whole prefix and canvas, or reuses an immutable causal
//! prefix. Every canvas pass starts from the same per-layer boundary states.
use crate::blocks::{Gate, Mixer, PrefixKv, SharedKv, ShortConvState};
use crate::{Forward, LoadError, Model, RopeScaling};
use kinetic_gguf::TensorSource;

pub const MODE_KEY: &str = "kinetic.diffusion.mode";
pub const MODE: &str = "lfm2-prefix-lm-v1";
pub const MAX_CANVAS: usize = 64;
pub const MAX_PREFIX: usize = 2048;

pub struct Diffusion<'a> {
    model: Model<'a>,
    mask: Vec<f32>,
}

enum PrefixMixer {
    Attention(PrefixKv),
    ShortConv(ShortConvState),
}

/// Prepared by and borrowed from one model; it cannot be applied to another.
/// Prefix states are immutable, including each convolution's boundary window.
pub struct Prefix<'m, 'a> {
    diffusion: &'m Diffusion<'a>,
    layers: Vec<PrefixMixer>,
}

impl Prefix<'_, '_> {
    /// Canvas-only work, starting at the prefix's absolute RoPE position.
    pub fn forward(&self, canvas: &[u32], masked: &[bool]) -> Result<Vec<f32>, LoadError> {
        let model = &self.diffusion.model;
        self.diffusion.validate_canvas(canvas, masked)?;
        let mut x = self.diffusion.embed_canvas(canvas, masked);
        for (layer, prefix) in model.layers.iter().zip(&self.layers) {
            layer.forward_with_routing_epsilon(&mut x, model.eps, 1e-6, |h| {
                match (&layer.mixer, prefix) {
                    (Mixer::Attention(a), PrefixMixer::Attention(kv)) => a.forward_canvas(h, kv),
                    (Mixer::ShortConv(c), PrefixMixer::ShortConv(state)) => {
                        c.forward(h, None, &mut state.clone())
                    }
                    _ => unreachable!("prefix prepared for the same model"),
                }
            });
        }
        model.norm_rows(&mut x, &model.output_norm);
        self.diffusion.logits(&x)
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub canvas: usize,
    pub steps: usize,
    pub max_new_tokens: usize,
    pub prefix_window: usize,
    pub pad: u32,
    pub eos: Vec<u32>,
}

impl Config {
    pub fn validate(&self, prompt: &[u32], vocab: usize) -> Result<(), LoadError> {
        let c = self;
        if !(1..=MAX_CANVAS).contains(&c.canvas)
            || !(1..=16).contains(&c.steps)
            || !(1..=256).contains(&c.max_new_tokens)
            || !(1..=MAX_PREFIX).contains(&c.prefix_window)
            || prompt.is_empty()
            || prompt.len().saturating_add(c.prefix_window) > MAX_PREFIX
            || c.pad as usize >= vocab
            || c.eos.iter().any(|&t| t as usize >= vocab)
            || prompt.iter().any(|&t| t as usize >= vocab)
            || vocab < 2
        {
            return Err(LoadError(
                "invalid CPU diffusion sampler settings or prompt".into(),
            ));
        }
        Ok(())
    }
}

/// Logits precede the pad-token ban. Non-masked rows have -inf confidence.
/// Commits are simultaneous; their order inside a pass is immaterial.
pub struct Step<'a> {
    pub block: usize,
    pub step: usize,
    pub prefix: &'a [u32],
    pub canvas_before: &'a [u32],
    pub masked_before: &'a [bool],
    pub logits: &'a [f32],
    pub candidates: &'a [u32],
    pub log_confidence: &'a [f32],
    pub committed: &'a [usize],
    pub canvas_after: &'a [u32],
}

#[derive(Debug)]
pub struct Generated {
    pub tokens: Vec<u32>,
    pub forward_calls: usize,
    pub prefix_builds: usize,
}

impl<'a> Diffusion<'a> {
    pub fn load(file: &'a dyn TensorSource, mask: Vec<f32>) -> Result<Self, LoadError> {
        if file.header().architecture() != Some("lfm2moe")
            || file.header().get(MODE_KEY).and_then(|v| v.as_str()) != Some(MODE)
        {
            return Err(LoadError(format!(
                "CPU diffusion requires lfm2moe with {MODE_KEY}={MODE}"
            )));
        }
        let model = crate::build(file, &RopeScaling::default())?;
        if mask.len() != model.n_embd() || mask.iter().any(|v| !v.is_finite()) {
            return Err(LoadError(
                "mask embedding must be finite and match the model width".into(),
            ));
        }
        if model.mtp.is_some() || model.per_layer.is_some() {
            return Err(LoadError(
                "diffusion does not support MTP or per-layer inputs".into(),
            ));
        }
        for layer in &model.layers {
            match &layer.mixer {
                Mixer::ShortConv(_) => {}
                Mixer::Attention(a)
                    if a.window.is_none()
                        && a.folding.is_none()
                        && matches!(a.gate, Gate::None)
                        && matches!(a.shared_kv, SharedKv::No) => {}
                _ => return Err(LoadError("unsupported mixer in CPU diffusion".into())),
            }
        }
        Ok(Self { model, mask })
    }

    pub fn n_vocab(&self) -> usize {
        self.model.n_vocab()
    }
    pub fn n_embd(&self) -> usize {
        self.model.n_embd()
    }

    fn validate_prefix(&self, prefix: &[u32]) -> Result<(), LoadError> {
        if prefix.is_empty()
            || prefix.len() > MAX_PREFIX
            || prefix.iter().any(|&t| t as usize >= self.n_vocab())
        {
            return Err(LoadError(
                "invalid diffusion prefix dimensions or token IDs".into(),
            ));
        }
        Ok(())
    }

    fn validate_canvas(&self, canvas: &[u32], masked: &[bool]) -> Result<(), LoadError> {
        if canvas.is_empty()
            || canvas.len() > MAX_CANVAS
            || canvas.len() != masked.len()
            || canvas.iter().any(|&t| t as usize >= self.n_vocab())
        {
            return Err(LoadError(
                "invalid diffusion canvas, mask or token IDs".into(),
            ));
        }
        Ok(())
    }

    fn embed_canvas(&self, canvas: &[u32], masked: &[bool]) -> Vec<f32> {
        let mut x = self.model.embed(canvas);
        for (row, &mask) in x.chunks_exact_mut(self.n_embd()).zip(masked) {
            if mask {
                row.copy_from_slice(&self.mask);
            }
        }
        x
    }

    fn logits(&self, normalized: &[f32]) -> Result<Vec<f32>, LoadError> {
        let logits = self.model.head(normalized);
        if logits.iter().any(|x| !x.is_finite()) {
            return Err(LoadError(
                "diffusion forward produced non-finite logits".into(),
            ));
        }
        Ok(logits)
    }

    /// Run the causal prefix once, saving rotated KV and conv boundary states.
    pub fn prepare_prefix(&self, prefix: &[u32]) -> Result<Prefix<'_, 'a>, LoadError> {
        self.validate_prefix(prefix)?;
        let mut x = self.model.embed(prefix);
        let mut layers = Vec::with_capacity(self.model.layers.len());
        for layer in &self.model.layers {
            layer.forward_with_routing_epsilon(&mut x, self.model.eps, 1e-6, |h| {
                match &layer.mixer {
                    Mixer::Attention(a) => {
                        let (kv, mixed) = a.prepare_prefix(h);
                        layers.push(PrefixMixer::Attention(kv));
                        mixed
                    }
                    Mixer::ShortConv(c) => {
                        let mut state = c.new_state();
                        let mixed = c.forward(h, None, &mut state);
                        layers.push(PrefixMixer::ShortConv(state));
                        mixed
                    }
                    _ => unreachable!("validated at load"),
                }
            });
        }
        Ok(Prefix {
            diffusion: self,
            layers,
        })
    }

    /// Canvas logits and normalized hidden rows for the whole sequence.
    pub fn forward(
        &self,
        prefix: &[u32],
        canvas: &[u32],
        masked: &[bool],
    ) -> Result<Forward, LoadError> {
        self.validate_prefix(prefix)?;
        self.validate_canvas(canvas, masked)?;
        let d = self.n_embd();
        let mut x = self.model.embed(prefix);
        x.extend(self.embed_canvas(canvas, masked));
        for layer in &self.model.layers {
            layer.forward_with_routing_epsilon(&mut x, self.model.eps, 1e-6, |h| {
                match &layer.mixer {
                    Mixer::Attention(a) => a.forward_prefix_lm(h, prefix.len()),
                    Mixer::ShortConv(c) => c.forward(h, None, &mut c.new_state()),
                    _ => unreachable!("validated at load"),
                }
            });
        }
        self.model.norm_rows(&mut x, &self.model.output_norm);
        let logits = self.logits(&x[prefix.len() * d..])?;
        Ok(Forward { logits, hidden: x })
    }

    pub fn generate(
        &self,
        prompt: &[u32],
        config: &Config,
        trace: impl FnMut(&Step<'_>) -> Result<(), LoadError>,
    ) -> Result<Generated, LoadError> {
        generate(
            prompt,
            self.n_vocab(),
            config,
            |_, p, c, m| self.forward(p, c, m).map(|f| f.logits),
            trace,
        )
    }

    pub fn generate_reusing_prefix(
        &self,
        prompt: &[u32],
        config: &Config,
        trace: impl FnMut(&Step<'_>) -> Result<(), LoadError>,
    ) -> Result<Generated, LoadError> {
        let mut cached = None;
        let mut cached_block = None;
        let mut builds = 0;
        let mut result = generate(
            prompt,
            self.n_vocab(),
            config,
            |block, prefix, canvas, masked| {
                if cached_block != Some(block) {
                    cached = Some(self.prepare_prefix(prefix)?);
                    cached_block = Some(block);
                    builds += 1;
                }
                cached.as_ref().unwrap().forward(canvas, masked)
            },
            trace,
        )?;
        result.prefix_builds = builds;
        Ok(result)
    }
}

fn generate(
    prompt: &[u32],
    vocab: usize,
    c: &Config,
    mut forward: impl FnMut(usize, &[u32], &[u32], &[bool]) -> Result<Vec<f32>, LoadError>,
    mut trace: impl FnMut(&Step<'_>) -> Result<(), LoadError>,
) -> Result<Generated, LoadError> {
    c.validate(prompt, vocab)?;
    let mut output = Vec::new();
    let mut calls = 0;
    let mut block = 0;
    while output.len() < c.max_new_tokens {
        let n = c.canvas.min(c.max_new_tokens - output.len());
        let prefix: Vec<u32> = prompt
            .iter()
            .chain(&output[output.len().saturating_sub(c.prefix_window)..])
            .copied()
            .collect();
        let mut canvas = vec![c.pad; n];
        let mut masked = vec![true; n];
        for step in 0..c.steps {
            let logits = forward(block, &prefix, &canvas, &masked)?;
            if logits.len() != n * vocab || logits.iter().any(|v| !v.is_finite()) {
                return Err(LoadError("invalid diffusion logits".into()));
            }
            calls += 1;
            let mut candidates = Vec::with_capacity(n);
            let mut confidence = Vec::with_capacity(n);
            for (row, &active) in logits.chunks_exact(vocab).zip(&masked) {
                let mut best = if c.pad == 0 { 1 } else { 0 };
                for (id, &value) in row.iter().enumerate() {
                    if id != c.pad as usize && value > row[best] {
                        best = id;
                    }
                }
                let sum: f32 = row
                    .iter()
                    .enumerate()
                    .filter(|(id, _)| *id != c.pad as usize)
                    .map(|(_, &v)| (v - row[best]).exp())
                    .sum();
                candidates.push(best as u32);
                confidence.push(if active { -sum.ln() } else { f32::NEG_INFINITY });
            }
            let before = canvas.clone();
            let mask_before = masked.clone();
            let mut committed: Vec<usize> = (0..n).filter(|&i| masked[i]).collect();
            if step + 1 != c.steps {
                let remaining = n * (c.steps - step - 1) / c.steps;
                let count = committed
                    .len()
                    .checked_sub(remaining)
                    .ok_or_else(|| LoadError("invalid commit schedule".into()))?;
                committed.sort_by(|&a, &b| confidence[b].total_cmp(&confidence[a]).then(a.cmp(&b)));
                committed.truncate(count);
            }
            for &i in &committed {
                canvas[i] = candidates[i];
                masked[i] = false;
            }
            trace(&Step {
                block,
                step,
                prefix: &prefix,
                canvas_before: &before,
                masked_before: &mask_before,
                logits: &logits,
                candidates: &candidates,
                log_confidence: &confidence,
                committed: &committed,
                canvas_after: &canvas,
            })?;
            if canvas
                .iter()
                .enumerate()
                .any(|(i, t)| !masked[i] && c.eos.contains(t) && masked[..i].iter().all(|&m| !m))
            {
                break;
            }
        }
        if let Some(end) = canvas.iter().position(|t| c.eos.contains(t)) {
            output.extend_from_slice(&canvas[..end]);
            break;
        }
        output.extend(canvas);
        block += 1;
    }
    Ok(Generated {
        tokens: output,
        forward_calls: calls,
        prefix_builds: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> Config {
        Config {
            canvas: 4,
            steps: 3,
            max_new_tokens: 9,
            prefix_window: 2,
            pad: 0,
            eos: vec![],
        }
    }

    #[test]
    fn fixed_schedule_history_and_short_final_canvas() {
        let mut traces = Vec::new();
        let result = generate(
            &[1, 2, 3],
            4,
            &config(),
            |_, _, canvas, _| Ok([0., 1., 2., 3.].repeat(canvas.len())),
            |s| {
                traces.push((s.block, s.step, s.prefix.to_vec(), s.committed.to_vec()));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(result.tokens, vec![3; 9]);
        assert_eq!(result.forward_calls, 9);
        assert_eq!(traces[0].3, [0, 1]);
        assert_eq!(traces[1].3, [2]);
        assert_eq!(traces[2].3, [3]);
        assert_eq!(traces[3].2, [1, 2, 3, 3, 3]);
        assert_eq!(traces[6].3, [0]);
        assert!(traces[7].3.is_empty());
    }

    #[test]
    fn eos_waits_for_every_preceding_position() {
        let mut cfg = config();
        cfg.eos = vec![2];
        cfg.max_new_tokens = 4;
        let mut commits = Vec::new();
        let out = generate(
            &[1],
            4,
            &cfg,
            |_, _, canvas, _| {
                let rows = [
                    [0., 1., -9., 1.1],
                    [0., 1., -9., 2.],
                    [0., -9., 9., -9.],
                    [0., 1., -9., 1.1],
                ];
                Ok(rows[..canvas.len()].iter().flatten().copied().collect())
            },
            |s| {
                commits.push(s.committed.to_vec());
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(commits, [vec![2, 1], vec![0]]);
        assert_eq!(out.tokens, [3, 3]);
        assert_eq!(out.forward_calls, 2);
    }

    #[test]
    fn rejects_invalid_dimensions_and_nonfinite_logits() {
        let mut c = config();
        c.steps = 0;
        assert!(generate(&[1], 4, &c, |_, _, _, _| unreachable!(), |_| Ok(())).is_err());
        assert!(generate(
            &[1],
            4,
            &config(),
            |_, _, c, _| Ok(vec![f32::NAN; c.len() * 4]),
            |_| Ok(())
        )
        .is_err());
    }
}
