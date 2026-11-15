//! A terminal conversation over the fixed CPU diffusion sampler. Only a
//! committed contiguous prefix is streamed: canvas guesses never leak as text.
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use clap::Args;
use kinetic_gguf::GgufFile;
use kinetic_models::diffusion::{Config, Diffusion, Step, MAX_PREFIX};
use kinetic_models::LoadError;
use kinetic_server::ChatTemplate;
use kinetic_tokenizer::Tokenizer;
use rustyline::{error::ReadlineError, DefaultEditor};
use serde_json::{json, Value};

use crate::render::Markdown;
use crate::Result;

const FALLBACK: &str =
    "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}assistant: ";
const HELP: &str = "/clear — новый диалог   /settings — настройки   /exit — выход\n/canvas 16|32|64   /steps 1..16   /tokens 1..256   /reuse on|off\n/system ТЕКСТ — задать инструкции и начать заново   /raw — Markdown вкл/выкл\nCtrl+C — остановить ответ; Ctrl+D — выйти. Стрелки вверх/вниз — история ввода.";

#[derive(Args)]
pub struct ChatArgs {
    /// Explicitly marked lfm2moe diffusion GGUF (always CPU)
    model: PathBuf,
    #[arg(long)]
    mask_embedding: PathBuf,
    #[arg(long, default_value_t = 16)]
    canvas: usize,
    #[arg(long, default_value_t = 4)]
    steps: usize,
    #[arg(short = 'n', long, default_value_t = 64)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 128)]
    prefix_window: usize,
    /// Use full recomputation as the reference instead of prefix reuse
    #[arg(long)]
    no_prefix_reuse: bool,
    #[arg(long)]
    system: Option<String>,
    /// Plain text: no ANSI, animation or Markdown formatting
    #[arg(long)]
    plain: bool,
    /// Keep colors and Markdown but disable animation
    #[arg(long)]
    no_animation: bool,
    /// Delay per displayed character; changes presentation only (0..30 ms)
    #[arg(long, default_value_t = 2)]
    pace_ms: u64,
    #[arg(short = 't', long)]
    pub threads: Option<usize>,
}

fn clean(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_control() || matches!(c, '\n' | '\t'))
        .collect()
}

fn opening(system: &Option<String>) -> Vec<Value> {
    system
        .iter()
        .map(|s| json!({"role":"system", "content":s}))
        .collect()
}

/// Drop only old complete user/assistant pairs, never the current message or system.
fn fit_history(
    messages: &mut Vec<Value>,
    limit: usize,
    encode: impl Fn(&[Value]) -> Result<Vec<u32>>,
) -> Result<(Vec<u32>, bool)> {
    let mut trimmed = false;
    loop {
        let prompt = encode(messages)?;
        if prompt.len() <= limit {
            return Ok((prompt, trimmed));
        }
        let first = usize::from(messages.first().is_some_and(|m| m["role"] == "system"));
        if messages.len() < first + 3 {
            return Err(format!("Сообщение слишком длинное: {} токенов, доступно {limit}. Укоротите сообщение или /system.", prompt.len()).into());
        }
        messages.drain(first..first + 2);
        trimmed = true;
    }
}

/// Number of final tokens that can be shown, before any hole or committed EOS.
fn stable_end(s: &Step<'_>, eos: &[u32]) -> usize {
    let mut masked = s.masked_before.to_vec();
    for &i in s.committed {
        masked[i] = false;
    }
    s.canvas_after
        .iter()
        .zip(masked)
        .position(|(t, mask)| mask || eos.contains(t))
        .unwrap_or(s.canvas_after.len())
}

enum Event {
    Progress(String),
    Text(String),
}

struct Answer {
    animated: bool,
    pace: Duration,
    renderer: Markdown,
    started: bool,
    frame: usize,
    progress: String,
    showing: bool,
}

impl Answer {
    fn tick(&mut self) -> io::Result<()> {
        if !self.animated || self.started {
            return Ok(());
        }
        const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let mut out = io::stdout().lock();
        write!(
            out,
            "\r\x1b[2K\x1b[38;2;177;155;231m  {}  {}\x1b[0m",
            FRAMES[self.frame % FRAMES.len()],
            self.progress
        )?;
        out.flush()?;
        self.frame += 1;
        self.showing = true;
        Ok(())
    }
    fn clear(&mut self, out: &mut impl Write) -> io::Result<()> {
        if self.showing {
            write!(out, "\r\x1b[2K")?;
            self.showing = false;
        }
        Ok(())
    }
    fn text(&mut self, text: &str, stop: &AtomicBool) -> io::Result<()> {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        let text = clean(text);
        if text.is_empty() {
            return Ok(());
        }
        let mut out = io::stdout().lock();
        self.clear(&mut out)?;
        self.started = true;
        if self.animated && !self.pace.is_zero() {
            for ch in text.chars() {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                self.renderer.write(&mut out, ch.encode_utf8(&mut [0; 4]))?;
                out.flush()?;
                std::thread::sleep(self.pace);
            }
        } else {
            self.renderer.write(&mut out, &text)?;
            out.flush()?;
        }
        Ok(())
    }
    fn finish(&mut self) -> io::Result<()> {
        let mut out = io::stdout().lock();
        self.clear(&mut out)?;
        self.renderer.finish(&mut out)?;
        if !self.started {
            write!(out, "(пустой ответ)")?;
        }
        writeln!(out)?;
        out.flush()
    }
}

fn answer(
    model: &Diffusion<'_>,
    tokenizer: &Tokenizer,
    prompt: &[u32],
    config: &Config,
    reuse: bool,
    ui: &mut Answer,
    stop: &AtomicBool,
) -> Result<kinetic_models::diffusion::Generated> {
    std::thread::scope(|scope| {
        let (tx, rx) = mpsc::channel();
        let worker = scope.spawn(move || {
            let mut decoder = tokenizer.stream();
            let mut block = None;
            let mut emitted = 0;
            let mut trace = |s: &Step<'_>| -> std::result::Result<(), LoadError> {
                if stop.load(Ordering::Relaxed) {
                    return Err(LoadError("chat interrupted".into()));
                }
                if block != Some(s.block) {
                    block = Some(s.block);
                    emitted = 0;
                }
                let filled = s.masked_before.iter().filter(|&&m| !m).count() + s.committed.len();
                let bars = 16 * filled / s.canvas_after.len();
                let progress = format!(
                    "блок {} · шаг {}/{}  [{}{}] {}/{}",
                    s.block + 1,
                    s.step + 1,
                    config.steps,
                    "━".repeat(bars),
                    "·".repeat(16 - bars),
                    filled,
                    s.canvas_after.len()
                );
                tx.send(Event::Progress(progress))
                    .map_err(|_| LoadError("chat output closed".into()))?;
                let end = stable_end(s, &config.eos);
                if end < emitted {
                    return Err(LoadError("committed canvas prefix shrank".into()));
                }
                let mut text = String::new();
                for &token in &s.canvas_after[emitted..end] {
                    text += &decoder.push(token);
                }
                emitted = end;
                if !text.is_empty() {
                    tx.send(Event::Text(text))
                        .map_err(|_| LoadError("chat output closed".into()))?;
                }
                Ok(())
            };
            let result = if reuse {
                model.generate_reusing_prefix(prompt, config, &mut trace)
            } else {
                model.generate(prompt, config, &mut trace)
            };
            if result.is_ok() {
                let _ = tx.send(Event::Text(decoder.finish()));
            }
            result
        });
        let display = (|| -> io::Result<()> {
            loop {
                match rx.recv_timeout(Duration::from_millis(70)) {
                    Ok(Event::Progress(progress)) => {
                        ui.progress = progress;
                        ui.tick()?;
                    }
                    Ok(Event::Text(text)) => ui.text(&text, stop)?,
                    Err(mpsc::RecvTimeoutError::Timeout) => ui.tick()?,
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
            ui.finish()
        })();
        if display.is_err() {
            stop.store(true, Ordering::Relaxed);
        }
        drop(rx);
        let result = worker.join().map_err(|_| "diffusion worker panicked")?;
        display?;
        if stop.load(Ordering::Relaxed) {
            return Err("chat interrupted".into());
        }
        Ok(result?)
    })
}

pub fn run(args: ChatArgs) -> Result<()> {
    if args.pace_ms > 30 {
        return Err("pace-ms must be in 0..30".into());
    }
    let file = GgufFile::open(&args.model)?;
    let model = Diffusion::load(&file, crate::diffusion::read_mask(&args.mask_embedding)?)?;
    let tokenizer = Tokenizer::from_gguf(file.header())?;
    let pad = file
        .header()
        .get("tokenizer.ggml.padding_token_id")
        .and_then(|v| v.as_u64())
        .and_then(|v| u32::try_from(v).ok())
        .ok_or("diffusion needs a pad token")?;
    let mut config = Config {
        canvas: args.canvas,
        steps: args.steps,
        max_new_tokens: args.max_new_tokens,
        prefix_window: args.prefix_window,
        pad,
        eos: tokenizer.eos().into_iter().collect(),
    };
    config.validate(&[pad], model.n_vocab())?;
    let source = file
        .header()
        .get("tokenizer.chat_template")
        .and_then(|v| v.as_str());
    let template = ChatTemplate::new(source.unwrap_or(FALLBACK))?;
    let interactive = io::stdin().is_terminal() && io::stdout().is_terminal();
    let color = interactive
        && !args.plain
        && std::env::var_os("NO_COLOR").is_none()
        && crate::chat::enable_escapes();
    let paint = |code: &str, text: &str| {
        if color {
            format!("\x1b[{code}m{text}\x1b[0m")
        } else {
            text.to_owned()
        }
    };
    println!(
        "\n{}",
        paint("1;38;2;192;167;245", "  KINETIC  ·  DIFFUSION CHAT")
    );
    println!(
        "{}",
        paint(
            "38;2;152;145;166",
            &format!(
                "  CPU · {} · fixed greedy · /help",
                if args.no_prefix_reuse {
                    "полный пересчёт"
                } else {
                    "prefix reuse"
                }
            )
        )
    );
    let name = file
        .header()
        .get("general.name")
        .and_then(|v| v.as_str())
        .unwrap_or("LFM2 diffusion");
    println!("  {}", clean(name));
    if name.to_lowercase().contains("fixture") || name.to_lowercase().contains("not trained") {
        println!(
            "{}",
            paint(
                "38;2;242;184;112",
                "  ТЕСТОВАЯ НЕОБУЧЕННАЯ МОДЕЛЬ — текст случайный, это демонстрация интерфейса."
            )
        );
    }
    println!("  /help — команды · /clear — новый диалог · /exit — выход\n");
    let stop = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&stop);
    ctrlc::set_handler(move || flag.store(true, Ordering::Relaxed))?;
    let mut editor = if interactive && !args.plain {
        Some(DefaultEditor::new()?)
    } else {
        None
    };
    let mut system = args.system;
    let mut messages = opening(&system);
    let mut reuse = !args.no_prefix_reuse;
    let mut raw = args.plain;
    loop {
        let line = if let Some(editor) = &mut editor {
            match editor.readline("Вы › ") {
                Ok(line) => {
                    let _ = editor.add_history_entry(&line);
                    line
                }
                Err(ReadlineError::Interrupted) => {
                    println!("Для выхода: /exit");
                    continue;
                }
                Err(ReadlineError::Eof) => break,
                Err(e) => return Err(e.into()),
            }
        } else {
            if interactive {
                print!("Вы > ");
                io::stdout().flush()?;
            }
            let mut line = String::new();
            if io::stdin().read_line(&mut line)? == 0 {
                break;
            }
            line
        };
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        match text {
            "/exit" | "/quit" => break,
            "/help" => {
                println!("{HELP}\n");
                continue;
            }
            "/clear" => {
                messages = opening(&system);
                println!("Диалог очищен.\n");
                continue;
            }
            "/raw" => {
                raw = !raw;
                println!("Markdown: {}\n", if raw { "выкл" } else { "вкл" });
                continue;
            }
            "/settings" => {
                println!(
                    "canvas={} · steps={} · tokens={} · prefix_window={} · reuse={}\n",
                    config.canvas, config.steps, config.max_new_tokens, config.prefix_window, reuse
                );
                continue;
            }
            _ => {}
        }
        let (command, value) = text.split_once(char::is_whitespace).unwrap_or((text, ""));
        if command == "/system" {
            system = (!value.trim().is_empty()).then(|| value.trim().to_owned());
            messages = opening(&system);
            println!("Инструкции обновлены. Новый диалог.\n");
            continue;
        }
        if command == "/reuse" {
            match value.trim() {
                "on" => reuse = true,
                "off" => reuse = false,
                _ => {
                    println!("Используйте /reuse on или /reuse off\n");
                    continue;
                }
            }
            println!("Prefix reuse: {}\n", if reuse { "вкл" } else { "выкл" });
            continue;
        }
        if matches!(command, "/canvas" | "/steps" | "/tokens") {
            let mut proposed = config.clone();
            let valid = value.trim().parse::<usize>().ok().is_some_and(|n| {
                match command {
                    "/canvas" => proposed.canvas = n,
                    "/steps" => proposed.steps = n,
                    _ => proposed.max_new_tokens = n,
                }
                proposed.validate(&[pad], model.n_vocab()).is_ok()
            });
            if valid {
                config = proposed;
                println!("Настройки обновлены.\n");
            } else {
                println!("Допустимо: /canvas 1..64, /steps 1..16, /tokens 1..256\n");
            }
            continue;
        }
        if text.starts_with('/') {
            println!("Неизвестная команда. /help — список команд.\n");
            continue;
        }
        let mut proposed = messages.clone();
        proposed.push(json!({"role":"user", "content":text}));
        let fitted = fit_history(
            &mut proposed,
            MAX_PREFIX - config.prefix_window,
            |history| {
                let rendered = template.render(json!({"messages":history, "add_generation_prompt":true,
                "bos_token":tokenizer.bos().map(|t| tokenizer.decode(&[t])).unwrap_or_default(),
                "eos_token":tokenizer.eos().map(|t| tokenizer.decode(&[t])).unwrap_or_default(), "enable_thinking":false}))?;
                Ok(tokenizer.encode(&rendered, true))
            },
        );
        let (prompt, trimmed) = match fitted {
            Ok(p) => p,
            Err(e) => {
                println!("{e}\n");
                continue;
            }
        };
        if trimmed {
            println!("Контекст сокращён: сохранены последние сообщения.");
        }
        config.validate(&prompt, model.n_vocab())?;
        println!("{}", paint("1;38;2;137;211;195", "Kinetic ›"));
        stop.store(false, Ordering::Relaxed);
        let start = Instant::now();
        let mut ui = Answer {
            animated: color && !args.no_animation,
            pace: Duration::from_millis(args.pace_ms),
            renderer: Markdown::new(color && !raw),
            started: false,
            frame: 0,
            progress: "подготовка префикса…".into(),
            showing: false,
        };
        let generated = answer(&model, &tokenizer, &prompt, &config, reuse, &mut ui, &stop);
        match generated {
            Ok(g) => {
                proposed.push(json!({"role":"assistant", "content":tokenizer.decode(&g.tokens)}));
                messages = proposed;
                println!(
                    "{}\n",
                    paint(
                        "38;2;152;145;166",
                        &format!(
                            "  {} токенов · {} проходов · {:.2} с",
                            g.tokens.len(),
                            g.forward_calls,
                            start.elapsed().as_secs_f64()
                        )
                    )
                );
            }
            Err(_) if stop.load(Ordering::Relaxed) => println!("Ответ остановлен.\n"),
            Err(e) => {
                println!("Ошибка генерации: {e}\n");
            }
        }
    }
    println!("До встречи.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stream_stops_at_holes_and_eos_even_if_later_positions_are_committed() {
        let before = [true, true, true, true];
        let after = [11, 12, 2, 13];
        let mut s = Step {
            block: 0,
            step: 0,
            prefix: &[1],
            canvas_before: &[0; 4],
            masked_before: &before,
            logits: &[],
            candidates: &[],
            log_confidence: &[],
            committed: &[1, 2, 3],
            canvas_after: &after,
        };
        assert_eq!(stable_end(&s, &[2]), 0);
        s.committed = &[0, 1, 2, 3];
        assert_eq!(stable_end(&s, &[2]), 2);
        assert_eq!(stable_end(&s, &[]), 4);
    }
    #[test]
    fn trimming_keeps_system_and_current_user_without_cutting_pairs() {
        let mut messages = vec![
            json!({"role":"system","content":"s"}),
            json!({"role":"user","content":"old"}),
            json!({"role":"assistant","content":"old"}),
            json!({"role":"user","content":"new"}),
        ];
        let (prompt, trimmed) = fit_history(&mut messages, 2, |m| Ok(vec![1; m.len()])).unwrap();
        assert!(trimmed);
        assert_eq!(prompt.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["content"], "new");
        assert!(fit_history(&mut messages, 1, |m| Ok(vec![1; m.len()])).is_err());
    }
    #[test]
    fn terminal_controls_are_removed_but_markdown_and_unicode_survive() {
        assert_eq!(
            clean("\x1b[2JПривет\r\x08\x07\n\t**тест**"),
            "[2JПривет\n\t**тест**"
        );
    }
}
