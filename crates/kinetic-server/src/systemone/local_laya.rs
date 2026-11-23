//! Runs exported Laya checkpoints in-process through the upstream Rust engine.

use laya::{Answer, LayaEngine, Question, QuestionSet};
use serde_json::{json, Map, Value};

use super::Failure;

pub(super) fn decide(engine: &LayaEngine, model: &str, request: &Value) -> Result<Value, Failure> {
    if request.get("assets").is_some() || request.get("images").is_some() {
        return Err(Failure::validation(
            &["assets"],
            "this Laya model accepts only text",
            "value_error",
        ));
    }
    let questions = request["questions"]
        .as_object()
        .expect("validated questions");
    let mut set = QuestionSet::new();
    for (id, question) in questions {
        let invalid = |e: laya::LayaError| {
            Failure::validation(&["questions", id], e.to_string(), "value_error")
        };
        let instructions = &question["instructions"];
        let criteria = &question["criteria"];
        let format = match question["type"].as_str() {
            Some("noul") => Question::noul(
                instructions.clone(),
                criteria["false"].clone(),
                criteria["true"].clone(),
            ),
            Some("choice") => Question::choice(
                instructions.clone(),
                criteria.as_object().expect("validated choices").clone(),
            )
            .map_err(invalid)?,
            Some("score") => Question::score(
                instructions.clone(),
                criteria.as_array().expect("validated rubric").clone(),
            )
            .map_err(invalid)?,
            _ => {
                return Err(Failure::validation(
                    &["questions", id, "type"],
                    "this Laya model accepts choice, score and noul",
                    "value_error",
                ))
            }
        };
        set.insert(id, format).map_err(invalid)?;
    }
    let result = engine
        .predict(request["state"].clone(), &set)
        .map_err(|error| {
            eprintln!("kinetic: Laya model {model:?}: {error}");
            Failure::new(500, "Laya inference failed")
        })?;
    if result.state_usage().is_some_and(|usage| usage.truncated) {
        return Err(Failure::new(
            400,
            "max_tokens_exceeded: state exceeds the Laya context",
        ));
    }
    let answers: Map<_, _> = result
        .iter()
        .map(|(id, answer)| {
            let value = match answer {
                Answer::Noul(noul) => json!({"type": "noul", "noul": noul.probability}),
                Answer::Choice(choice) => {
                    let probabilities: Map<_, _> = choice
                        .probabilities
                        .iter()
                        .map(|(key, probability)| (key.clone(), json!(probability)))
                        .collect();
                    json!({
                        "type": "choice",
                        "choice": choice.choice,
                        "confidence": answer.confidence(),
                        "probabilities": probabilities,
                    })
                }
                Answer::Score(score) => {
                    let probabilities: Map<_, _> = score
                        .probabilities
                        .iter()
                        .enumerate()
                        .map(|(index, probability)| (index.to_string(), json!(probability)))
                        .collect();
                    let legend: Map<_, _> = questions[id]["criteria"]
                        .as_array()
                        .expect("validated rubric")
                        .iter()
                        .enumerate()
                        .map(|(index, value)| (index.to_string(), value.clone()))
                        .collect();
                    json!({
                        "type": "score", "score": score.score, "confidence": answer.confidence(),
                        "legend": legend, "probabilities": probabilities,
                    })
                }
            };
            (id.to_owned(), value)
        })
        .collect();
    let usage = result.usage();
    Ok(json!({
        "model": model,
        "answers": answers,
        "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
    }))
}
