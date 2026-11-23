//! TypeSafe's System One wire protocol over configured decision-model providers.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, value::RawValue, Value};
use url::Url;

use crate::api::ApiError;
use crate::http::{self, Head};

#[cfg(feature = "jeff-onnx")]
mod local_jeff;
#[cfg(feature = "laya-onnx")]
mod local_laya;
#[cfg(test)]
mod tests;

pub(crate) const MAX_BODY: usize = 32 << 20;
const MAX_NESTING: usize = 64;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    models: Vec<ModelConfig>,
    default_model: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelConfig {
    name: String,
    description: String,
    release_date: String,
    base_url: Option<String>,
    laya_onnx: Option<PathBuf>,
    jeff_onnx: Option<PathBuf>,
    jeff_encoder: Option<PathBuf>,
    jeff_isolate: Option<JeffIsolation>,
    upstream_model: Option<String>,
    api_key_env: Option<String>,
    #[serde(default)]
    supports_images: bool,
}

#[derive(Serialize)]
struct Metadata {
    name: String,
    description: String,
    release_date: String,
}

struct Provider {
    endpoint: Url,
    model: Option<String>,
    api_key: Option<String>,
    supports_images: bool,
}

enum Backend {
    Http(Provider),
    #[cfg(feature = "laya-onnx")]
    Laya(Box<laya::LayaEngine>),
    #[cfg(feature = "jeff-onnx")]
    Jeff(Box<local_jeff::JeffEngine>),
}

/// When Jeff puts questions in separate rows of the encoder batch.
#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum JeffIsolation {
    None,
    #[default]
    Nouls,
    All,
}

pub struct Server {
    providers: HashMap<String, Backend>,
    default_model: Option<String>,
    models: Vec<Metadata>,
    agent: ureq::Agent,
}

impl Server {
    /// Loads the catalog and credentials. Only operator-configured URLs are contacted.
    pub fn from_json(bytes: &[u8], base_dir: &Path) -> Result<Self, String> {
        let config: Config = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if config.models.is_empty() {
            return Err("configure at least one System One model".into());
        }
        let mut providers = HashMap::new();
        let mut models = Vec::new();
        for entry in config.models {
            if entry.name.trim().is_empty() || providers.contains_key(&entry.name) {
                return Err(format!(
                    "empty or duplicate System One model name: {:?}",
                    entry.name
                ));
            }
            if !date_format(&entry.release_date) {
                return Err(format!(
                    "release_date for {} must be YYYY-MM-DD",
                    entry.name
                ));
            }
            let backend = entry.backend(base_dir)?;
            models.push(Metadata {
                name: entry.name.clone(),
                description: entry.description,
                release_date: entry.release_date,
            });
            providers.insert(entry.name, backend);
        }
        if config
            .default_model
            .as_ref()
            .is_some_and(|name| !providers.contains_key(name))
        {
            return Err("default_model must name a model in the catalog".into());
        }
        let agent = ureq::AgentBuilder::new()
            .redirects(0)
            .timeout_connect(Duration::from_secs(5))
            .timeout(Duration::from_secs(120))
            .build();
        Ok(Self {
            providers,
            default_model: config.default_model,
            models,
            agent,
        })
    }

    pub(crate) fn route(&self, head: &Head, body: &[u8], out: &mut impl Write) {
        let result = match head.path.as_str() {
            "/health" | "/v1/health" => method(head, "GET").map(|()| {
                write_json(out, 200, &json!({"status": "ok"}));
            }),
            "/v1/models" => method(head, "GET").map(|()| {
                write_json(out, 200, &json!({"models": self.models}));
            }),
            "/v1/systemone" => method(head, "POST").and_then(|()| self.decide(head, body, out)),
            _ => Err(Failure::new(404, "unknown endpoint")),
        };
        if let Err(error) = result {
            write_json(out, error.status, &json!({"detail": error.detail}));
        }
    }

    fn decide(&self, head: &Head, body: &[u8], out: &mut impl Write) -> Result<(), Failure> {
        // A browser can send multipart forms across origins without preflight.
        // Avoid forwarding those submissions, even to an unauthenticated listener.
        if let Some(origin) = head.header("origin") {
            let same_origin = Url::parse(origin).ok().is_some_and(|url| {
                let authority = &url[url::Position::BeforeHost..url::Position::AfterPort];
                head.header("host")
                    .is_some_and(|host| authority.eq_ignore_ascii_case(host))
            });
            if !same_origin || head.header("sec-fetch-site") == Some("cross-site") {
                return Err(Failure::new(403, "cross-origin requests are not accepted"));
            }
        }
        let body = decompress(head, body)?;
        let content_type = head.header("content-type").unwrap_or("application/json");
        let media = content_type.split(';').next().unwrap_or("").trim();
        let part = if media.eq_ignore_ascii_case("application/json") {
            0..body.len()
        } else if media.eq_ignore_ascii_case("multipart/form-data") {
            request_part(content_type, &body)?
        } else {
            return Err(Failure::new(
                415,
                "expected application/json or multipart/form-data",
            ));
        };
        let request: Value = serde_json::from_slice(&body[part.clone()])
            .map_err(|e| Failure::new(400, format!("invalid JSON: {e}")))?;
        let object = request
            .as_object()
            .ok_or_else(|| Failure::validation(&[], "expected an object", "dict_type"))?;
        let model = match object.get("model") {
            Some(Value::String(name)) => name.as_str(),
            None => self
                .default_model
                .as_deref()
                .ok_or_else(|| Failure::missing(&["model"]))?,
            Some(_) => {
                return Err(Failure::validation(
                    &["model"],
                    "expected a model name",
                    "string_type",
                ))
            }
        };
        let backend = self.providers.get(model).ok_or_else(|| {
            Failure::validation(
                &["model"],
                format!("unknown model {model:?}"),
                "value_error",
            )
        })?;
        let questions = validate(&request, head)?;
        match backend {
            Backend::Http(provider) => {
                let uses_images = request.get("assets").is_some()
                    || media.eq_ignore_ascii_case("multipart/form-data")
                    || questions.values().any(|q| q["type"] == "box");
                if uses_images && !provider.supports_images {
                    return Err(Failure::validation(
                        &["model"],
                        "this model does not accept image assets or box questions",
                        "value_error",
                    ));
                }
                let target = provider.model.as_deref().unwrap_or(model);
                let body = if request["model"].as_str() == Some(target) {
                    body
                } else {
                    replace_model(&body, part, target)?
                };
                provider.forward(&self.agent, head, content_type, &body, questions, out)
            }
            #[cfg(feature = "laya-onnx")]
            Backend::Laya(engine) => {
                if media.eq_ignore_ascii_case("multipart/form-data") {
                    return Err(Failure::new(415, "this Laya model accepts only text"));
                }
                let response = local_laya::decide(engine, model, &request)?;
                write_json(out, 200, &response);
                Ok(())
            }
            #[cfg(feature = "jeff-onnx")]
            Backend::Jeff(engine) => {
                if media.eq_ignore_ascii_case("multipart/form-data") {
                    return Err(Failure::new(415, "this Jeff model accepts only text"));
                }
                let response = engine.decide(model, &request, questions)?;
                write_json(out, 200, &response);
                Ok(())
            }
        }
    }
}

impl ModelConfig {
    fn backend(&self, base_dir: &Path) -> Result<Backend, String> {
        let local =
            self.upstream_model.is_none() && self.api_key_env.is_none() && !self.supports_images;
        let jeff_options = self.jeff_isolate.is_some() || self.jeff_encoder.is_some();
        match (&self.base_url, &self.laya_onnx, &self.jeff_onnx) {
            (Some(url), None, None) if !jeff_options => Provider::new(url, self).map(Backend::Http),
            (None, Some(dir), None) if local && !jeff_options => {
                let path = base_dir.join(dir);
                #[cfg(feature = "laya-onnx")]
                {
                    let engine = laya::LayaEngine::from_directory(&path)
                        .map_err(|e| format!("cannot load Laya at {}: {e}", path.display()))?;
                    Ok(Backend::Laya(Box::new(engine)))
                }
                #[cfg(not(feature = "laya-onnx"))]
                Err(format!(
                    "Laya model {} needs a Kinetic build with the laya-onnx feature",
                    path.display()
                ))
            }
            (None, None, Some(dir)) if local => {
                let path = base_dir.join(dir);
                #[cfg(feature = "jeff-onnx")]
                {
                    let engine = local_jeff::JeffEngine::load(
                        &path, self.jeff_encoder.as_deref(), self.jeff_isolate.unwrap_or_default(),
                    )
                        .map_err(|e| format!("cannot load Jeff at {}: {e}", path.display()))?;
                    Ok(Backend::Jeff(Box::new(engine)))
                }
                #[cfg(not(feature = "jeff-onnx"))]
                Err(format!(
                    "Jeff model {} needs a Kinetic build with the jeff-onnx feature",
                    path.display()
                ))
            }
            _ => Err(format!(
                "configure one backend (base_url, laya_onnx or jeff_onnx) and options for that backend for {}",
                self.name
            )),
        }
    }
}

impl Provider {
    fn new(base_url: &str, entry: &ModelConfig) -> Result<Self, String> {
        let mut endpoint =
            Url::parse(base_url).map_err(|_| format!("invalid base_url for {}", entry.name))?;
        if !matches!(endpoint.scheme(), "http" | "https")
            || endpoint.host_str().is_none()
            || !endpoint.username().is_empty()
            || endpoint.password().is_some()
            || endpoint.query().is_some()
            || endpoint.fragment().is_some()
        {
            return Err(format!("base_url for {} must be an HTTP(S) base URL without credentials or query parameters", entry.name));
        }
        let path = endpoint.path().trim_end_matches('/');
        let suffix = if path.ends_with("/v1") {
            "/systemone"
        } else {
            "/v1/systemone"
        };
        endpoint.set_path(&format!("{path}{suffix}"));
        let api_key = entry
            .api_key_env
            .as_ref()
            .map(|name| {
                let value = std::env::var(name)
                    .map_err(|_| format!("missing environment variable {name}"))?;
                if value.is_empty() || !value.bytes().all(|b| b.is_ascii_graphic()) {
                    return Err(format!("invalid bearer key in environment variable {name}"));
                }
                Ok(value)
            })
            .transpose()?;
        Ok(Self {
            endpoint,
            model: entry.upstream_model.clone(),
            api_key,
            supports_images: entry.supports_images,
        })
    }

    fn forward(
        &self,
        agent: &ureq::Agent,
        head: &Head,
        content_type: &str,
        body: &[u8],
        questions: &serde_json::Map<String, Value>,
        out: &mut impl Write,
    ) -> Result<(), Failure> {
        let mut request = agent
            .post(self.endpoint.as_str())
            .set("Content-Type", content_type)
            .set("Accept", "application/json");
        if let Some(key) = &self.api_key {
            request = request.set("Authorization", &format!("Bearer {key}"));
        }
        for key in [
            "TypeSafe-Danger-Zone-Images",
            "TypeSafe-Danger-Zone-Long-Context",
        ] {
            if let Some(value) = head.header(key) {
                request = request.set(key, value);
            }
        }
        let response = match request.send_bytes(body) {
            Ok(response) | Err(ureq::Error::Status(_, response)) => response,
            Err(_) => return Err(Failure::new(502, "cannot reach decision model provider")),
        };
        let status = response.status();
        if !(status == 200 || (400..600).contains(&status)) {
            return Err(Failure::new(
                502,
                "unexpected status from decision model provider",
            ));
        }
        let headers: Vec<_> = [
            "Retry-After",
            "Retry-After-Ms",
            "X-TypeSafe-Request-Id",
            "X-Request-Id",
        ]
        .into_iter()
        .filter_map(|name| response.header(name).map(|value| (name, value.to_owned())))
        .filter(|(_, value)| !value.contains(['\r', '\n']))
        .collect();
        let mut body = Vec::new();
        response
            .into_reader()
            .take(MAX_BODY as u64 + 1)
            .read_to_end(&mut body)
            .map_err(|_| Failure::new(502, "cannot read response from decision model provider"))?;
        if body.len() > MAX_BODY {
            return Err(Failure::new(
                502,
                "decision model provider response is too large",
            ));
        }
        let headers: Vec<_> = headers
            .iter()
            .map(|&(name, ref value)| (name, value.as_str()))
            .collect();
        let body = if status == 200 {
            verify_response(&body, questions)?;
            body
        } else {
            provider_error(status, &body)
        };
        let _ = http::write_response_with_headers(out, status, "application/json", &headers, &body);
        Ok(())
    }
}

fn verify_response(body: &[u8], questions: &serde_json::Map<String, Value>) -> Result<(), Failure> {
    let response: Value = serde_json::from_slice(body)
        .map_err(|_| Failure::new(502, "decision model provider returned invalid JSON"))?;
    let matches_questions = response["answers"].as_object().is_some_and(|answers| {
        answers.len() == questions.len()
            && questions.iter().all(|(id, question)| {
                answers
                    .get(id)
                    .is_some_and(|answer| matches_answer(question, answer))
            })
    });
    if response["model"].is_string()
        && matches_questions
        && response["usage"]["input_tokens"].is_u64()
        && response["usage"]["output_tokens"].is_u64()
    {
        Ok(())
    } else {
        Err(Failure::new(
            502,
            "decision model provider returned an invalid System One response",
        ))
    }
}

fn matches_answer(question: &Value, answer: &Value) -> bool {
    if answer["type"] != question["type"] {
        return false;
    }
    let probability = |value: &Value| value.as_f64().is_some_and(|n| (0.0..=1.0).contains(&n));
    let probabilities = |value: &Value| {
        value
            .as_object()
            .is_some_and(|map| map.values().all(probability))
    };
    match answer["type"].as_str() {
        Some("noul") => probability(&answer["noul"]),
        Some("choice") => {
            answer["choice"].is_string()
                && probability(&answer["confidence"])
                && probabilities(&answer["probabilities"])
        }
        Some("score") => {
            answer["score"].is_number()
                && answer["legend"].is_object()
                && probability(&answer["confidence"])
                && probabilities(&answer["probabilities"])
        }
        Some("box") => {
            ["top", "bottom", "left", "right"]
                .iter()
                .all(|key| probability(&answer["box"][*key]))
                && answer
                    .get("confidence")
                    .is_some_and(|value| value.is_null() || probability(value))
        }
        _ => false,
    }
}

fn provider_error(status: u16, body: &[u8]) -> Vec<u8> {
    if let Ok(mut value) = serde_json::from_slice::<Value>(body) {
        if status == 422 {
            if let Some(detail) = value["detail"].as_str() {
                let error = Failure::validation(&[], detail, "value_error");
                value["detail"] = error.detail;
            }
        }
        if matches!(value["detail"], Value::Array(_) | Value::String(_)) {
            return value.to_string().into_bytes();
        }
    }
    let error = if status == 422 {
        Failure::validation(
            &[],
            "decision model provider rejected the request",
            "value_error",
        )
    } else {
        Failure::new(status, "decision model provider rejected the request")
    };
    json!({"detail": error.detail}).to_string().into_bytes()
}

fn method(head: &Head, expected: &str) -> Result<(), Failure> {
    if head.method == expected {
        Ok(())
    } else {
        Err(Failure::new(
            405,
            format!("use {expected} for {}", head.path),
        ))
    }
}

fn date_format(date: &str) -> bool {
    date.len() == 10
        && date.bytes().enumerate().all(|(i, b)| match i {
            4 | 7 => b == b'-',
            _ => b.is_ascii_digit(),
        })
}

fn decompress<'a>(head: &Head, body: &'a [u8]) -> Result<Cow<'a, [u8]>, Failure> {
    let mut encodings = head.values("content-encoding");
    let encoding = encodings.next().unwrap_or("identity");
    if encodings.next().is_some() {
        return Err(Failure::new(415, "expected one content encoding"));
    }
    let invalid = |_| Failure::new(400, "malformed compressed body");
    let mut decoded = Vec::new();
    match encoding.to_ascii_lowercase().as_str() {
        "identity" => return Ok(Cow::Borrowed(body)),
        "gzip" | "x-gzip" => flate2::read::MultiGzDecoder::new(body)
            .take(MAX_BODY as u64 + 1)
            .read_to_end(&mut decoded)
            .map_err(invalid)?,
        "zstd" => {
            let mut decoder = zstd::stream::read::Decoder::new(body).map_err(invalid)?;
            decoder.window_log_max(23).map_err(invalid)?;
            decoder
                .take(MAX_BODY as u64 + 1)
                .read_to_end(&mut decoded)
                .map_err(invalid)?
        }
        _ => return Err(Failure::new(415, "unsupported content encoding")),
    };
    if decoded.len() > MAX_BODY {
        return Err(Failure::new(
            413,
            "the decompressed request body is too large",
        ));
    }
    Ok(Cow::Owned(decoded))
}

fn parameter(text: &str, key: &str) -> Option<String> {
    text.split(';')
        .skip(1)
        .filter_map(|part| part.trim().split_once('='))
        .find(|(name, _)| name.trim().eq_ignore_ascii_case(key))
        .map(|(_, value)| {
            let value = value.trim();
            value
                .strip_prefix('"')
                .and_then(|v| v.strip_suffix('"'))
                .unwrap_or(value)
                .to_owned()
        })
}

/// Only the first part is needed for routing. The provider parses and validates
/// image parts against its own capabilities; their bytes remain untouched.
fn request_part(content_type: &str, body: &[u8]) -> Result<Range<usize>, Failure> {
    let bad = || {
        Failure::new(
            400,
            "multipart body must start with a JSON part named request",
        )
    };
    let boundary = parameter(content_type, "boundary")
        .filter(|v| {
            (1..=70).contains(&v.len()) && v.bytes().all(|b| b.is_ascii_graphic() || b == b' ')
        })
        .ok_or_else(bad)?;
    let start = format!("--{boundary}\r\n");
    let rest = body.strip_prefix(start.as_bytes()).ok_or_else(bad)?;
    let end_headers = find(rest, b"\r\n\r\n")
        .filter(|&i| i <= http::MAX_HEAD)
        .ok_or_else(bad)?;
    let headers = std::str::from_utf8(&rest[..end_headers]).map_err(|_| bad())?;
    let name = headers
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find(|(name, _)| name.eq_ignore_ascii_case("content-disposition"))
        .and_then(|(_, value)| parameter(value, "name"));
    if name.as_deref() != Some("request") {
        return Err(bad());
    }
    let begin = start.len() + end_headers + 4;
    let separator = format!("\r\n--{boundary}");
    let end = find(&body[begin..], separator.as_bytes()).ok_or_else(bad)?;
    Ok(begin..begin + end)
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn replace_model<'a>(
    body: &[u8],
    part: Range<usize>,
    model: &str,
) -> Result<Cow<'a, [u8]>, Failure> {
    // The order of state fields and options can affect decision models.
    // Copy those JSON values verbatim when editing the top-level model field.
    let invalid = |_| Failure::new(400, "cannot rewrite the request model");
    let mut fields: BTreeMap<String, &RawValue> =
        serde_json::from_slice(&body[part.clone()]).map_err(invalid)?;
    let model = serde_json::value::to_raw_value(model).map_err(invalid)?;
    fields.insert("model".into(), &model);
    let json = serde_json::to_string(&fields).map_err(invalid)?;
    let len = body.len() - part.len() + json.len();
    if len > MAX_BODY {
        return Err(Failure::new(
            413,
            "the request body is too large after applying the model alias",
        ));
    }
    let mut buffer = Vec::with_capacity(len);
    buffer.extend_from_slice(&body[..part.start]);
    buffer.extend_from_slice(json.as_bytes());
    buffer.extend_from_slice(&body[part.end..]);
    Ok(Cow::Owned(buffer))
}

struct Failure {
    status: u16,
    detail: Value,
}

impl Failure {
    fn new(status: u16, detail: impl Into<String>) -> Self {
        Self {
            status,
            detail: json!(detail.into()),
        }
    }

    fn validation(path: &[&str], detail: impl Into<String>, kind: &str) -> Self {
        let loc: Vec<_> = std::iter::once("body")
            .chain(path.iter().copied())
            .collect();
        Self {
            status: 422,
            detail: json!([{"loc": loc, "msg": detail.into(), "type": kind}]),
        }
    }

    fn missing(path: &[&str]) -> Self {
        Self::validation(path, "Field required", "missing")
    }
}

fn write_json(out: &mut impl Write, status: u16, value: &Value) {
    let _ = http::write_response(
        out,
        status,
        "application/json",
        value.to_string().as_bytes(),
    );
}

pub(crate) fn send_error(out: &mut impl Write, error: &ApiError) {
    write_json(out, error.status, &json!({"detail": error.message}));
}

fn validate<'a>(
    request: &'a Value,
    head: &Head,
) -> Result<&'a serde_json::Map<String, Value>, Failure> {
    validate_content(
        request
            .get("state")
            .ok_or_else(|| Failure::missing(&["state"]))?,
        &["state"],
        false,
    )?;
    let questions = request
        .get("questions")
        .ok_or_else(|| Failure::missing(&["questions"]))?
        .as_object()
        .filter(|map| !map.is_empty())
        .ok_or_else(|| {
            Failure::validation(
                &["questions"],
                "expected at least one named question",
                "value_error",
            )
        })?;
    let image_mode = head.header("typesafe-danger-zone-images") == Some("true");
    if let Some(assets) = request.get("assets") {
        if !image_mode || assets.as_object().is_none_or(serde_json::Map::is_empty) {
            return Err(Failure::validation(
                &["assets"],
                "expected image assets and TypeSafe-Danger-Zone-Images: true",
                "value_error",
            ));
        }
    }
    for (id, question) in questions {
        if !question.is_object() {
            return Err(Failure::validation(
                &["questions", id],
                "expected an object",
                "dict_type",
            ));
        }
        if let Some(value) = question.get("instructions") {
            validate_content(value, &["questions", id, "instructions"], true)?;
        }
        let path = ["questions", id, "criteria"];
        let criteria = || {
            question
                .get("criteria")
                .ok_or_else(|| Failure::missing(&path))
        };
        let invalid = |message| Failure::validation(&path, message, "value_error");
        match question.get("type").and_then(Value::as_str) {
            Some("choice") => {
                let options = criteria()?
                    .as_object()
                    .filter(|map| (1..=255).contains(&map.len()))
                    .ok_or_else(|| invalid("expected 1 to 255 named choices"))?;
                depth(criteria()?, 0)?;
                for (key, value) in options {
                    validate_content(value, &["questions", id, "criteria", key], true)?;
                }
            }
            Some("score") => {
                let levels = criteria()?
                    .as_array()
                    .filter(|levels| (1..=10).contains(&levels.len()))
                    .ok_or_else(|| invalid("expected 1 to 10 score levels"))?;
                depth(criteria()?, 0)?;
                for level in levels {
                    validate_content(level, &path, false)?;
                }
            }
            Some("noul") => {
                if let Some(criteria) = question.get("criteria").filter(|v| !v.is_null()) {
                    let values = criteria
                        .as_object()
                        .ok_or_else(|| invalid("expected an object for noul criteria"))?;
                    depth(criteria, 0)?;
                    for key in ["false", "true"] {
                        if let Some(value) = values.get(key) {
                            validate_content(value, &["questions", id, "criteria", key], true)?;
                        }
                    }
                }
            }
            Some("box") if image_mode => {}
            Some("box") => {
                return Err(Failure::validation(
                    &["questions", id, "type"],
                    "box requires TypeSafe-Danger-Zone-Images: true",
                    "value_error",
                ))
            }
            _ => {
                return Err(Failure::validation(
                    &["questions", id, "type"],
                    "expected choice, score, noul or box",
                    "union_tag_invalid",
                ))
            }
        }
    }
    Ok(questions)
}

fn validate_content(value: &Value, path: &[&str], nullable: bool) -> Result<(), Failure> {
    if matches!(value, Value::String(_) | Value::Array(_) | Value::Object(_))
        || (nullable && value.is_null())
    {
        depth(value, 0)
    } else {
        Err(Failure::validation(
            path,
            "expected a string, object or array",
            "value_error",
        ))
    }
}

fn depth(value: &Value, current: usize) -> Result<(), Failure> {
    if current >= MAX_NESTING && (value.is_object() || value.is_array()) {
        return Err(Failure::new(
            400,
            "state, instructions and criteria may have at most 64 nested levels",
        ));
    }
    match value {
        Value::Object(map) => map.values().try_for_each(|value| depth(value, current + 1)),
        Value::Array(values) => values
            .iter()
            .try_for_each(|value| depth(value, current + 1)),
        _ => Ok(()),
    }
}
