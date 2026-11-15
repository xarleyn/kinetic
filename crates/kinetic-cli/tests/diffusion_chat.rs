//! Piped console chat on the existing synthetic FP32 fixture. Compares every
//! displayed answer with independent one-shot diffusion, including turn history.
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde_json::Value;

fn inputs() -> (PathBuf, PathBuf) {
    let model =
        PathBuf::from(std::env::var("KINETIC_TEST_DIFFUSION_GGUF").expect("set fixture path"));
    let mask = model.parent().unwrap().join("mask.safetensors");
    (model, mask)
}

fn one_shot(prompt: &str) -> String {
    let (model, mask) = inputs();
    let output = Command::new(env!("CARGO_BIN_EXE_kinetic"))
        .arg("diffuse")
        .arg(model)
        .arg("--mask-embedding")
        .arg(mask)
        .args([
            "--prompt",
            prompt,
            "--canvas",
            "16",
            "--steps",
            "4",
            "-n",
            "32",
            "--prefix-window",
            "128",
            "-t",
            "6",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice::<Value>(&output.stdout).unwrap()["text"]
        .as_str()
        .unwrap()
        .to_owned()
}

fn displayed(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_control() || matches!(c, '\n' | '\t'))
        .collect()
}

#[test]
#[ignore = "needs KINETIC_TEST_DIFFUSION_GGUF and sibling mask.safetensors"]
fn chat_streams_complete_answers_remembers_history_and_clear_resets_it() {
    let first = one_shot("user: Hi\nassistant: ");
    let second = one_shot(&format!(
        "user: Hi\nassistant: {first}\nuser: How?\nassistant: "
    ));
    assert!(!displayed(&first).is_empty());
    assert!(!displayed(&second).is_empty());
    let (model, mask) = inputs();
    let mut process = Command::new(env!("CARGO_BIN_EXE_kinetic"))
        .arg("diffuse-chat")
        .arg(model)
        .arg("--mask-embedding")
        .arg(mask)
        .args([
            "--plain",
            "--canvas",
            "16",
            "--steps",
            "4",
            "-n",
            "32",
            "--prefix-window",
            "128",
            "-t",
            "6",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    process
        .stdin
        .take()
        .unwrap()
        .write_all(b"Hi\nHow?\n/clear\nHi\n/canvas 0\n/settings\n/exit\n")
        .unwrap();
    let output = process.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(!text.contains('\x1b'), "piped chat must never emit ANSI");
    assert!(text.contains("ТЕСТОВАЯ НЕОБУЧЕННАЯ МОДЕЛЬ"));
    assert_eq!(text.matches("Kinetic ›").count(), 3);
    assert_eq!(
        text.matches(&displayed(&first)).count(),
        2,
        "clear must reproduce first answer"
    );
    assert!(
        text.contains(&displayed(&second)),
        "second answer must include first turn's raw history"
    );
    assert!(
        text.contains("canvas=16 · steps=4 · tokens=32"),
        "invalid /canvas must preserve settings"
    );
    assert!(text.contains("До встречи."));
}
