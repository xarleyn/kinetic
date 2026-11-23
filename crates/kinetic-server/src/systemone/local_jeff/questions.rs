//! The question prompts and probability normalization from Jeff's System One adapter.

use std::collections::HashSet;

use serde_json::{json, Map, Value};

use super::Failure;

const TEMPERATURE: f64 = 3.2;

pub(super) enum Kind<'a> {
    Noul,
    Choice(&'a Map<String, Value>),
    Score(&'a [Value]),
}

pub(super) struct Group<'a> {
    pub id: &'a str,
    pub kind: Kind<'a>,
    pub name: String,
    pub labels: Vec<String>,
}

impl<'a> Group<'a> {
    pub fn new(id: &'a str, question: &'a Value) -> Result<Self, Failure> {
        let criteria = &question["criteria"];
        let (kind, mut labels) = match question["type"].as_str() {
            Some("noul") => (
                Kind::Noul,
                vec![
                    describe("yes", &criteria["true"]),
                    describe("no", &criteria["false"]),
                ],
            ),
            Some("choice") => {
                let choices = criteria.as_object().expect("validated choices");
                let labels = choices
                    .iter()
                    .map(|(key, detail)| describe(key, detail))
                    .collect();
                (Kind::Choice(choices), labels)
            }
            Some("score") => {
                let levels = criteria.as_array().expect("validated rubric");
                let labels = levels
                    .iter()
                    .map(|level| state_text(level.get("what").unwrap_or(level)))
                    .collect();
                (Kind::Score(levels), labels)
            }
            _ => {
                return Err(Failure::validation(
                    &["questions", id, "type"],
                    "this Jeff model accepts choice, score and noul",
                    "value_error",
                ))
            }
        };
        let mut used = HashSet::new();
        for label in &mut labels {
            if used.insert(label.clone()) {
                continue;
            }
            for copy in 2.. {
                let candidate = format!("{label} #{copy}");
                if used.insert(candidate.clone()) {
                    *label = candidate;
                    break;
                }
            }
        }
        let mut name = state_text(&question["instructions"]);
        if name.is_empty() {
            name = id.to_owned();
        }
        Ok(Self {
            id,
            kind,
            name,
            labels,
        })
    }

    pub fn answer(&self, scores: &[f64]) -> Value {
        let probabilities = normalize(scores, TEMPERATURE);
        let max = probabilities.iter().copied().fold(0.0_f64, f64::max);
        let confidence = if probabilities.len() == 1 {
            1.0
        } else {
            let chance = 1.0 / probabilities.len() as f64;
            rounded((max - chance) / (1.0 - chance))
        };
        match self.kind {
            Kind::Noul => json!({"type": "noul", "noul": rounded(probabilities[0])}),
            Kind::Choice(choices) => {
                let best = probabilities
                    .iter()
                    .position(|&p| p == max)
                    .expect("a choice has options");
                let distribution: Map<_, _> = choices
                    .keys()
                    .zip(probabilities)
                    .map(|(key, p)| (key.clone(), json!(rounded(p))))
                    .collect();
                json!({
                    "type": "choice", "choice": choices.keys().nth(best), "confidence": confidence,
                    "probabilities": distribution,
                })
            }
            Kind::Score(levels) => {
                // Jeff reports the raw expected score, while its displayed
                // probabilities and confidence use the fitted temperature.
                let score: f64 = normalize(scores, 1.0)
                    .iter()
                    .enumerate()
                    .map(|(i, p)| i as f64 * p)
                    .sum();
                let legend: Map<_, _> = levels
                    .iter()
                    .enumerate()
                    .map(|(i, level)| (i.to_string(), level.clone()))
                    .collect();
                let distribution: Map<_, _> = probabilities
                    .into_iter()
                    .enumerate()
                    .map(|(i, p)| (i.to_string(), json!(rounded(p))))
                    .collect();
                json!({
                    "type": "score", "score": rounded(score), "confidence": confidence,
                    "legend": legend, "probabilities": distribution,
                })
            }
        }
    }
}

fn describe(label: &str, detail: &Value) -> String {
    let detail = state_text(detail);
    if detail.is_empty() {
        label.to_owned()
    } else {
        format!("{label}: {detail}")
    }
}

pub(super) fn state_text(value: &Value) -> String {
    match value {
        Value::Object(fields) => fields
            .iter()
            .map(|(key, value)| format!("{key}: {}", scalar(value)))
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Array(fields) => fields.iter().map(scalar).collect::<Vec<_>>().join("\n"),
        _ => scalar(value),
    }
}

fn scalar(value: &Value) -> String {
    match value {
        Value::String(text) => text.to_owned(),
        Value::Null => String::new(),
        _ => value.to_string(),
    }
}

fn normalize(scores: &[f64], temperature: f64) -> Vec<f64> {
    let mut values: Vec<_> = scores
        .iter()
        .map(|score| score.powf(1.0 / temperature))
        .collect();
    let sum: f64 = values.iter().sum();
    if sum <= 1e-9 {
        values.fill(1.0 / scores.len() as f64);
    } else {
        values.iter_mut().for_each(|value| *value /= sum);
    }
    values
}

fn rounded(value: f64) -> f64 {
    (value * 10_000.0).round_ties_even() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_order_and_rubric_shape() {
        assert_eq!(
            state_text(&json!({"z": 1, "a": [{"nested": true}]})),
            "z: 1\na: [{\"nested\":true}]"
        );
        let score =
            json!({"type":"score", "criteria":[{"what":"low","examples":["later"]},"low","high"]});
        let group = Group::new("urgent", &score).ok().unwrap();
        assert_eq!(group.labels, ["low", "low #2", "high"]);
        let answer = group.answer(&[0.1, 0.1, 0.8]);
        assert_eq!(answer["score"], 1.7);
        assert_eq!(answer["legend"]["0"], score["criteria"][0]);
    }

    #[test]
    fn choice_ties_use_the_first_label_and_noul_uses_yes_first() {
        let choice = json!({"type":"choice", "criteria":{"z":null,"a":"details"}});
        let group = Group::new("team", &choice).ok().unwrap();
        assert_eq!(group.labels, ["z", "a: details"]);
        assert_eq!(group.answer(&[0.0, 0.0])["choice"], "z");
        let noul = json!({"type":"noul","criteria":{"true":"requested","false":"cancelled"}});
        let group = Group::new("refund", &noul).ok().unwrap();
        assert_eq!(group.labels, ["yes: requested", "no: cancelled"]);
        assert!(group.answer(&[0.8, 0.2])["noul"].as_f64().unwrap() > 0.5);
    }
}
