//! The GLiFormer text classifier used by logan-markewich/jeff.

mod questions;

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use regex::Regex;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokenizers::Tokenizer;

use super::{Failure, JeffIsolation};
use questions::{Group, Kind};

const BATCH_SIZE: usize = 16;
const PROJECTION: &str = "heads.classification.anchor_modeling.proj";

type LoadResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Deserialize)]
struct Checkpoint {
    model_type: String,
    hidden_size: usize,
    max_len: usize,
    words_splitter_type: String,
    seq_token: String,
    sep_token: String,
    cat_parent_token: Option<String>,
    parent_token: String,
    cat_token: String,
    labels_encoder: Option<String>,
    classification_config: Classifier,
}

#[derive(Deserialize)]
struct Classifier {
    anchor_mode: String,
    anchor_modeling: String,
    anchor_normalization: String,
    anchor_layer: Option<Value>,
    anchor_refinement: Option<Value>,
    anchor_refine_layers: usize,
    pooling_type: String,
    scorer_type: String,
    parent_token_index: u32,
    cat_token_index: u32,
    embed_parent_token: bool,
    embed_cat_token: bool,
}

impl Checkpoint {
    fn validate(&self) -> LoadResult<()> {
        let head = &self.classification_config;
        if self.model_type != "gliformer-layout"
            || self.words_splitter_type != "whitespace"
            || self.labels_encoder.is_some()
            || self.hidden_size == 0
            || self.hidden_size > 4096
            || self.max_len < 4
            || self.max_len > 16384
            || head.anchor_mode != "parent"
            || head.anchor_modeling != "linear"
            || head.anchor_normalization != "none"
            || head.anchor_layer.is_some()
            || head.anchor_refinement.is_some()
            || head.anchor_refine_layers != 0
            || head.pooling_type != "cls"
            || head.scorer_type != "dot"
            || !head.embed_parent_token
            || !head.embed_cat_token
        {
            return Err("unsupported GLiFormer classifier configuration".into());
        }
        Ok(())
    }

    fn parent(&self) -> &str {
        self.cat_parent_token
            .as_deref()
            .unwrap_or(&self.parent_token)
    }
}

pub(super) struct JeffEngine {
    isolate: JeffIsolation,
    config: Checkpoint,
    tokenizer: Tokenizer,
    splitter: Regex,
    pad: u32,
    head: Projection,
    encoder: Mutex<Session>,
}

impl JeffEngine {
    pub(super) fn load(
        dir: &Path,
        graph: Option<&Path>,
        isolate: JeffIsolation,
    ) -> LoadResult<Self> {
        let config: Checkpoint =
            serde_json::from_reader(File::open(dir.join("gliner_config.json"))?)?;
        config.validate()?;
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        let pad = tokenizer
            .get_padding()
            .map(|padding| padding.pad_id)
            .or_else(|| tokenizer.token_to_id("[PAD]"))
            .ok_or("tokenizer has no pad token")?;
        tokenizer.with_padding(None);
        let truncation = tokenizers::TruncationParams {
            max_length: config.max_len + 1,
            ..Default::default()
        };
        tokenizer.with_truncation(Some(truncation))?;
        if tokenizer.token_to_id(config.parent())
            != Some(config.classification_config.parent_token_index)
            || tokenizer.token_to_id(&config.cat_token)
                != Some(config.classification_config.cat_token_index)
        {
            return Err("GLiFormer classifier markers do not match the tokenizer".into());
        }

        let head = Projection::load(&dir.join("pytorch_model.bin"), config.hidden_size)?;
        let graph = graph.map_or_else(|| encoder_path(dir), |graph| dir.join(graph));
        let encoder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| e.to_string())?
            .commit_from_file(&graph)
            .map_err(|e| format!("cannot open {}: {e}", graph.display()))?;
        if encoder.inputs().len() != 2
            || !encoder
                .inputs()
                .iter()
                .any(|input| input.name() == "input_ids")
            || !encoder
                .inputs()
                .iter()
                .any(|input| input.name() == "attention_mask")
            || !encoder
                .outputs()
                .iter()
                .any(|output| output.name() == "token_embeds")
        {
            return Err(
                "expected the Jeff ONNX encoder with input_ids, attention_mask and token_embeds"
                    .into(),
            );
        }
        Ok(Self {
            isolate,
            config,
            tokenizer,
            pad,
            head,
            encoder: Mutex::new(encoder),
            // Python's Unicode \w is alphanumeric plus underscore.
            splitter: Regex::new(r"[\p{L}\p{N}_]+(?:[-_][\p{L}\p{N}_]+)*|[^\s]")?,
        })
    }

    pub(super) fn decide(
        &self,
        model: &str,
        request: &Value,
        questions: &Map<String, Value>,
    ) -> Result<Value, Failure> {
        if request.get("assets").is_some() || request.get("images").is_some() {
            return Err(Failure::validation(
                &["assets"],
                "this Jeff model accepts only text",
                "value_error",
            ));
        }
        let groups = questions
            .iter()
            .map(|(id, q)| Group::new(id, q))
            .collect::<Result<Vec<_>, _>>()?;
        let state = questions::state_text(&request["state"]);
        let tokens: Vec<_> = self
            .splitter
            .find_iter(&state)
            .take(self.config.max_len + 1)
            .map(|m| m.as_str())
            .collect();
        let mut rows = Vec::new();
        let mut shared = Vec::new();
        for (index, group) in groups.iter().enumerate() {
            let isolated = match self.isolate {
                JeffIsolation::None => false,
                JeffIsolation::Nouls => matches!(group.kind, Kind::Noul),
                JeffIsolation::All => true,
            };
            if isolated {
                rows.push(self.prepare(&tokens, &groups, &[index])?);
            } else {
                shared.push(index);
            }
        }
        if !shared.is_empty() {
            rows.push(self.prepare(&tokens, &groups, &shared)?);
        }
        let mut scores = vec![Vec::new(); groups.len()];
        let mut encoder = self
            .encoder
            .lock()
            .map_err(|_| Failure::new(500, "Jeff encoder is unavailable"))?;
        for batch in rows.chunks(BATCH_SIZE) {
            self.score_batch(&mut encoder, batch, &mut scores)
                .map_err(|error| {
                    eprintln!("kinetic: Jeff model {model:?}: {error}");
                    Failure::new(500, "Jeff inference failed")
                })?;
        }
        let answers: Map<_, _> = groups
            .into_iter()
            .zip(scores)
            .map(|(group, scores)| (group.id.to_owned(), group.answer(&scores)))
            .collect();
        let input_tokens: usize = rows.iter().map(|r| r.ids.len()).sum();
        // Jeff's System One API uses a nominal six output tokens per answer.
        let output_tokens = 6 * answers.len();
        Ok(json!({
            "model": model, "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }))
    }

    fn prepare(
        &self,
        state: &[&str],
        groups: &[Group<'_>],
        indices: &[usize],
    ) -> Result<Row, Failure> {
        let mut words = vec![self.config.seq_token.clone()];
        let mut positions = Vec::with_capacity(indices.len());
        for &index in indices {
            let group = &groups[index];
            let anchor = words.len();
            words.push(self.config.parent().to_owned());
            if !group.name.is_empty() {
                words.push(group.name.clone());
            }
            let labels = words.len();
            for label in &group.labels {
                words.push(format!("{} {label}", self.config.cat_token));
            }
            positions.push(Markers {
                group: index,
                anchor,
                labels: (labels..words.len()).collect(),
            });
            words.push(self.config.sep_token.clone());
        }
        words.push(self.config.sep_token.clone());
        if state.is_empty() {
            words.push("[PAD]".into());
        } else {
            words.extend(state.iter().map(|&text| text.to_owned()));
        }
        let encoding = self
            .tokenizer
            .encode(words.as_slice(), true)
            .map_err(|e| Failure::new(400, format!("cannot tokenize request: {e}")))?;
        let ids = encoding.get_ids();
        if ids.len() > self.config.max_len {
            return Err(Failure::new(
                400,
                "max_tokens_exceeded: request exceeds the Jeff context",
            ));
        }
        let mut starts = vec![None; words.len()];
        for (position, word) in encoding.get_word_ids().iter().enumerate() {
            if let Some(word) = word {
                starts[*word as usize].get_or_insert(position);
            }
        }
        let locate = |word: usize, id: u32| -> Result<usize, Failure> {
            starts[word]
                .filter(|&position| ids[position] == id)
                .ok_or_else(|| {
                    Failure::new(
                        400,
                        "cannot locate Jeff classification markers in the request",
                    )
                })
        };
        for markers in &mut positions {
            markers.anchor = locate(
                markers.anchor,
                self.config.classification_config.parent_token_index,
            )?;
            for mark in &mut markers.labels {
                *mark = locate(*mark, self.config.classification_config.cat_token_index)?;
            }
        }
        Ok(Row {
            ids: ids.iter().copied().map(i64::from).collect(),
            markers: positions,
        })
    }

    fn score_batch(
        &self,
        encoder: &mut Session,
        batch: &[Row],
        scores: &mut [Vec<f64>],
    ) -> LoadResult<()> {
        let seq = batch
            .iter()
            .map(|row| row.ids.len())
            .max()
            .ok_or("empty Jeff batch")?;
        let width = self.config.hidden_size;
        let mut input_ids = vec![i64::from(self.pad); batch.len() * seq];
        let mut attention = vec![0_i64; input_ids.len()];
        for (i, row) in batch.iter().enumerate() {
            let range = i * seq..i * seq + row.ids.len();
            input_ids[range.clone()].copy_from_slice(&row.ids);
            attention[range].fill(1);
        }
        let outputs = encoder.run(ort::inputs![
            "input_ids" => Tensor::from_array(([batch.len(), seq], input_ids))?,
            "attention_mask" => Tensor::from_array(([batch.len(), seq], attention))?,
        ])?;
        let (shape, data) = outputs
            .get("token_embeds")
            .ok_or("encoder omitted token_embeds")?
            .try_extract_tensor::<f32>()?;
        if shape.as_ref() != [batch.len() as i64, seq as i64, width as i64] {
            return Err(format!("unexpected Jeff encoder output shape {shape:?}").into());
        }
        for (row, output) in batch.iter().zip(data.chunks_exact(seq * width)) {
            let embedding = |position| &output[position * width..][..width];
            let (packed, bias) = self.head.for_cls(embedding(0));
            let (parent, child) = packed.split_at(width);
            for markers in &row.markers {
                let offset = kinetic_cpu::dot(parent, embedding(markers.anchor)) + bias;
                let result = &mut scores[markers.group];
                for &label in &markers.labels {
                    let logit = kinetic_cpu::dot(child, embedding(label)) + offset;
                    if !logit.is_finite() {
                        return Err("non-finite Jeff classification logit".into());
                    }
                    result.push(f64::from(kinetic_cpu::sigmoid(logit)));
                }
            }
        }
        Ok(())
    }
}

struct Row {
    ids: Vec<i64>,
    markers: Vec<Markers>,
}

struct Markers {
    group: usize,
    anchor: usize,
    labels: Vec<usize>,
}

fn encoder_path(dir: &Path) -> PathBuf {
    let quantized = dir.join("onnx/encoder.int8.onnx");
    if quantized.is_file() {
        quantized
    } else {
        dir.join("onnx/encoder.onnx")
    }
}

struct Projection {
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl Projection {
    fn load(path: &Path, width: usize) -> LoadResult<Self> {
        let file = File::open(path)?;
        // Model files are immutable while loading, as with Kinetic's GGUF mmap.
        let bytes = unsafe { memmap2::Mmap::map(&file)? };
        if let Some(order) = mlmf_pickle::read_zip_directory(&bytes)?
            .into_iter()
            .find(|entry| entry.name.ends_with("/byteorder"))
        {
            let start = usize::try_from(order.data_offset)?;
            if order.method != 0
                || order.size != 6
                || bytes.get(start..start.saturating_add(6)) != Some(b"little")
            {
                return Err("Jeff requires little-endian PyTorch weights".into());
            }
        }
        let archive = mlmf_pickle::parse_pth(&bytes, None)?;
        let tensor = |suffix: &str, shape: &[usize], strides: &[usize]| -> LoadResult<Vec<f32>> {
            let name = format!("{PROJECTION}.{suffix}");
            let (info, range) = archive
                .tensors
                .iter()
                .zip(&archive.storage_ranges)
                .find(|(info, _)| info.name == name)
                .ok_or_else(|| format!("missing tensor {name}"))?;
            if info.shape != shape
                || info.strides != strides
                || info.storage_class != "FloatStorage"
            {
                return Err(format!(
                    "expected a contiguous F32 tensor {name} with shape {shape:?}"
                )
                .into());
            }
            let range = range
                .as_ref()
                .ok_or_else(|| format!("compressed tensor {name} is unsupported"))?;
            let offset = usize::try_from(info.storage_offset)?
                .checked_mul(4)
                .ok_or("invalid tensor offset")?;
            let from = usize::try_from(range.start)?
                .checked_add(offset)
                .ok_or("invalid tensor offset")?;
            let limit = usize::try_from(range.end)?;
            let count: usize = shape.iter().product();
            let to = from
                .checked_add(count * 4)
                .filter(|&end| end <= limit)
                .ok_or("truncated tensor")?;
            let raw = bytes.get(from..to).ok_or("truncated tensor storage")?;
            Ok(raw
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect())
        };
        Ok(Self {
            weights: tensor("weight", &[width, 2 * width], &[2 * width, 1])?,
            bias: tensor("bias", &[width], &[1])?,
        })
    }

    /// GLiFormer scores dot(CLS, W × [parent; label] + b).
    /// Project CLS once through the transposed weights to reuse it for all labels.
    fn for_cls(&self, cls: &[f32]) -> (Vec<f32>, f32) {
        let mut packed = vec![0.0; 2 * cls.len()];
        let stride = packed.len();
        kinetic_cpu::axpy_rows(&mut packed, cls, &self.weights, stride);
        (packed, kinetic_cpu::dot(&self.bias, cls))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projected_cls_scores_the_parent_and_label_in_the_original_order() {
        let head = Projection {
            weights: vec![1., 2., 3., 4., 5., 6., 7., 8.],
            bias: vec![0.5, -0.5],
        };
        let (weights, bias) = head.for_cls(&[2., 3.]);
        assert_eq!(weights, [17., 22., 27., 32.]);
        assert_eq!(bias, -0.5);
    }

    #[test]
    #[ignore = "requires KINETIC_TEST_JEFF_DIR with GLiFormer Large v1 and its encoder export"]
    fn real_checkpoint_matches_the_reference_prompt_and_runs_the_head() {
        let path = std::env::var_os("KINETIC_TEST_JEFF_DIR").expect("set KINETIC_TEST_JEFF_DIR");
        let engine = JeffEngine::load(Path::new(&path), None, JeffIsolation::None).unwrap();
        let request = json!({
            "state": "customer support ticket about billing customer support ticket about billing",
            "questions": {
                "team": {"type":"choice","instructions":"Which team should handle this?",
                         "criteria":{"billing":null,"technical":null,"sales":null}},
                "frustration": {"type":"choice","instructions":"How frustrated is the customer?",
                                "criteria":{"calm":null,"annoyed":null,"angry":null}},
                "refund": {"type":"noul","instructions":"Does the customer request a refund?"}
            }
        });
        let questions = request["questions"].as_object().unwrap();
        let groups: Vec<_> = questions
            .iter()
            .map(|(id, q)| Group::new(id, q).ok().unwrap())
            .collect();
        let words: Vec<_> = request["state"]
            .as_str()
            .unwrap()
            .split_whitespace()
            .collect();
        let row = engine.prepare(&words, &groups, &[0, 1, 2]).ok().unwrap();
        // From Jeff's original tokenizer/collator with the same named groups.
        assert_eq!(
            row.ids,
            [
                1, 647, 106441, 592, 128002, 2597, 511, 403, 1997, 291, 302, 128003, 9833, 128003,
                2064, 128003, 1160, 2, 128002, 577, 8572, 269, 262, 1099, 302, 128003, 4847,
                128003, 16244, 128003, 5355, 2, 128002, 2467, 262, 1099, 1409, 266, 6257, 302,
                128003, 2489, 128003, 363, 2, 2, 1099, 523, 3316, 314, 9833, 1099, 523, 3316, 314,
                9833, 2,
            ]
        );
        let answer = engine.decide("jeff", &request, questions).ok().unwrap();
        assert_eq!(answer["usage"]["input_tokens"], row.ids.len());
        assert_eq!(answer["answers"]["team"]["type"], "choice");
        assert!(answer["answers"]["refund"]["noul"].is_number());
    }

    #[test]
    #[ignore = "requires KINETIC_TEST_JEFF_DIR with GLiFormer Large v1 and the FP32 encoder"]
    fn real_fp32_checkpoint_matches_jeff_answers() {
        let path = std::env::var_os("KINETIC_TEST_JEFF_DIR").expect("set KINETIC_TEST_JEFF_DIR");
        let mut engine = JeffEngine::load(
            Path::new(&path),
            Some(Path::new("onnx/encoder.onnx")),
            JeffIsolation::None,
        )
        .unwrap();
        let request = json!({
            "state":"I was charged twice. Please refund the extra charge.",
            "questions":{
                "team":{"type":"choice","instructions":"Which team should handle this?",
                        "criteria":{"billing":"Invoices and payments","tech":"Technical issues","other":null}},
                "urgent":{"type":"score","instructions":"How urgent?","criteria":["low","medium","high"]},
                "duplicate":{"type":"noul","instructions":"Did the user report a duplicate charge?"}
            }
        });
        let questions = request["questions"].as_object().unwrap();
        let result = engine.decide("jeff", &request, questions).ok().unwrap();
        assert_eq!(
            result["usage"],
            json!({"input_tokens":63,"output_tokens":18})
        );
        let answers = &result["answers"];
        assert_eq!(answers["team"]["choice"], "tech");
        assert_eq!(
            answers["urgent"]["legend"],
            json!({"0":"low","1":"medium","2":"high"})
        );
        for (value, expected) in [
            (&answers["team"]["probabilities"]["tech"], 0.4177),
            (&answers["urgent"]["score"], 0.9427),
            (&answers["duplicate"]["noul"], 0.7820),
        ] {
            assert!(
                (value.as_f64().unwrap() - expected).abs() <= 0.0002,
                "{value} vs {expected}"
            );
        }
        engine.isolate = JeffIsolation::Nouls;
        let isolated = engine.decide("jeff", &request, questions).ok().unwrap();
        assert!(isolated["usage"]["input_tokens"].as_u64().unwrap() > 63);
        assert!(isolated["answers"]["duplicate"]["noul"].is_number());
    }
}
