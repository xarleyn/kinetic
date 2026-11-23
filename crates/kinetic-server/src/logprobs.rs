//! Token log-probabilities in OpenAI's response shapes.

use serde_json::{json, Map, Value};

use crate::TokenLogprob;

/// Maps a visible prefix of decoded text back to the tokens that supplied it.
#[derive(Default)]
pub(crate) struct TextLogprobs {
    bytes: Vec<u8>,
    starts: Vec<usize>,
    visible: usize,
}

impl TextLogprobs {
    pub(crate) fn push(&mut self, piece: &[u8]) {
        self.starts.push(self.bytes.len());
        self.bytes.extend_from_slice(piece);
    }

    pub(crate) fn emit(&mut self, text: &str) -> usize {
        let mut left = text.len();
        if left != 0 {
            // A replacement character uses three bytes in text, but may
            // stand for fewer input bytes. Include UTF-8 lookahead so a
            // malformed sequence is not mistaken for a buffered tail.
            let pending = &self.bytes[self.visible..];
            for chunk in pending[..pending.len().min(left + 3)].utf8_chunks() {
                let valid = chunk.valid().len().min(left);
                self.visible += valid;
                left -= valid;
                if left == 0 {
                    break;
                }
                self.visible += chunk.invalid().len();
                left -= '\u{fffd}'.len_utf8();
                if left == 0 {
                    break;
                }
            }
        }
        // A stop can cut inside a token. Keep it if its first byte is visible.
        self.starts.partition_point(|&start| start < self.visible)
    }
}

/// `choices[].logprobs` of a chat completion or chunk.
pub fn chat<'t>(tokens: &[TokenLogprob], piece: impl Fn(u32) -> &'t [u8]) -> Value {
    let entry = |id: u32, logprob: f32| {
        let bytes = piece(id);
        json!({"token": String::from_utf8_lossy(bytes), "logprob": finite(logprob), "bytes": bytes})
    };
    let content: Vec<Value> = tokens
        .iter()
        .map(|t| {
            let mut e = entry(t.token, t.logprob);
            e["top_logprobs"] = t.top.iter().map(|&(id, lp)| entry(id, lp)).collect();
            e
        })
        .collect();
    json!({"content": content, "refusal": null})
}

/// `choices[].logprobs` of a legacy completion or chunk; `offset` is where
/// in the generated text, in characters, the first token starts, and moves
/// past the last.
pub fn completion<'t>(
    tokens: &[TokenLogprob],
    piece: impl Fn(u32) -> &'t [u8],
    offset: &mut usize,
) -> Value {
    let text = |id: u32| String::from_utf8_lossy(piece(id)).into_owned();
    let mut offsets = Vec::with_capacity(tokens.len());
    for t in tokens {
        offsets.push(*offset);
        *offset += text(t.token).chars().count();
    }
    let top: Vec<Value> = tokens
        .iter()
        .map(|t| {
            let alternatives: Map<String, Value> = t
                .top
                .iter()
                .map(|&(id, lp)| (text(id), finite(lp).into()))
                .collect();
            alternatives.into()
        })
        .collect();
    json!({
        "tokens": tokens.iter().map(|t| text(t.token)).collect::<Vec<_>>(),
        "token_logprobs": tokens.iter().map(|t| finite(t.logprob)).collect::<Vec<_>>(),
        "top_logprobs": top,
        "text_offset": offsets,
    })
}

/// JSON has no infinity; OpenAI reports impossible tokens as -9999.
fn finite(logprob: f32) -> f32 {
    logprob.max(-9999.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens() -> Vec<TokenLogprob> {
        vec![
            TokenLogprob {
                token: 0,
                logprob: -0.5,
                top: vec![(0, -0.5), (1, -1.0)],
            },
            TokenLogprob {
                token: 1,
                logprob: f32::NEG_INFINITY,
                top: Vec::new(),
            },
        ]
    }

    fn piece(id: u32) -> &'static [u8] {
        [b"Hi".as_slice(), "ś".as_bytes()][id as usize]
    }

    #[test]
    fn visible_prefix_can_end_inside_a_token() {
        let mut text = TextLogprobs::default();
        text.push(b"Hi");
        text.push(b" stop");
        text.push(b" here");
        assert_eq!(text.emit("Hi "), 2);
    }

    #[test]
    fn replacement_character_does_not_count_buffered_bytes() {
        let mut text = TextLogprobs::default();
        text.push(&[0xf0, 0x9f]);
        assert_eq!(text.emit(""), 0);
        text.push(b"A");
        text.push(b"B");
        assert_eq!(text.emit("\u{fffd}"), 1);
        assert_eq!(text.emit("A"), 2);
        assert_eq!(text.emit("B"), 3);
    }

    #[test]
    fn chat_shape() {
        let v = chat(&tokens(), piece);
        assert_eq!(v["content"][0]["token"], "Hi");
        assert_eq!(v["content"][0]["bytes"], json!([72, 105]));
        assert_eq!(v["content"][0]["top_logprobs"][1]["logprob"], -1.0);
        assert_eq!(v["content"][1]["logprob"], -9999.0);
        assert_eq!(v["refusal"], Value::Null);
    }

    #[test]
    fn completion_shape() {
        let mut offset = 3;
        let v = completion(&tokens(), piece, &mut offset);
        assert_eq!(v["tokens"], json!(["Hi", "ś"]));
        assert_eq!(v["token_logprobs"], json!([-0.5, -9999.0]));
        assert_eq!(v["top_logprobs"][0], json!({"Hi": -0.5, "ś": -1.0}));
        assert_eq!(v["text_offset"], json!([3, 5]));
        assert_eq!(offset, 6);
    }
}
