//! This is the spm command line utility.

use spm_core::{
    spm::{Context, Master, Mode, Worker},
    Args,
};

use anyhow::{anyhow, Result};
use base64::Engine;
use clap::Parser;
use std::{fs, io::Write, path::Path};
use tokio::io::{AsyncBufReadExt, BufReader};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectedModel {
    Qwen3Vl,
    Llama3,
}

/// （新增）根据图片扩展名推断 mime 类型。
///
/// 为什么要加：Qwen3-VL 的 `image_base64` part 允许携带 `media_type`；
/// 这能帮助服务端/模型更准确地理解图片格式（png/jpg/webp...）。
fn guess_media_type(path: &str) -> Option<String> {
    let ext = Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let mime = match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        _ => return None,
    };
    Some(mime.to_string())
}

/// （新增）把本地图片读出来并转成 `image_base64` 这个多模态输入 part。
///
/// 为什么要加：用户希望“命令行直接问图”，而不是手写一大坨 JSON + base64；
/// 这个函数把“读文件 + base64 编码 + 组装 ContentPart”封装起来。
fn image_part_from_path(path: &str) -> Result<spm_core::models::chat::ContentPart> {
    let bytes = fs::read(path).map_err(|e| anyhow!("can't read image {}: {e}", path))?;
    let data = base64::engine::general_purpose::STANDARD.encode(bytes);
    Ok(spm_core::models::chat::ContentPart::ImageBase64 {
        media_type: guess_media_type(path),
        data,
    })
}

/// （新增）生成一个 user message：既支持纯文本，也支持“文本 + 图片”的多模态 message。
///
/// 实现了什么：当传入 `image_path` 时，构造 OpenAI 风格的 `content: [ {text}, {image_base64} ]`；
/// 不传图片时则退化为普通文本消息，兼容纯文本模型/请求。
fn user_message_text_or_image(
    text: String,
    image_path: Option<&str>,
) -> Result<spm_core::models::chat::Message> {
    if let Some(path) = image_path {
        // Qwen3-VL 的官方/常见用法是“先给图，再提问”，因此把 image part 放在 text 前面。
        let parts = vec![
            image_part_from_path(path)?,
            spm_core::models::chat::ContentPart::Text { text },
        ];
        Ok(spm_core::models::chat::Message {
            role: spm_core::models::chat::MessageRole::User,
            content: spm_core::models::chat::MessageContent::Parts(parts),
        })
    } else {
        Ok(spm_core::models::chat::Message::user(text))
    }
}

/// （新增）把 `--api-client` 的输入统一规范成 base URL。
///
/// 为什么要加：用户可能传 `127.0.0.1:8082` / `http://127.0.0.1:8082/` 等各种形式；
/// 统一后拼接 `/api/v1/chat/completions` 更稳妥。
fn normalize_api_base(raw: &str) -> String {
    let mut s = raw.trim().to_string();
    if s.is_empty() {
        return s;
    }
    if !s.starts_with("http://") && !s.starts_with("https://") {
        s = format!("http://{s}");
    }
    s.trim_end_matches('/').to_string()  //把末尾的/去掉
}

/// Read an OpenAI-style SSE stream and print delta tokens to stdout as they arrive.
/// Returns the full assistant content and optional metrics when provided by the server.
async fn consume_sse_stream(
    mut resp: reqwest::Response,
    print_deltas: bool,
) -> Result<(String, Option<f64>, Option<f64>)> {
    let mut buf = String::new();
    let mut out = String::new();
    let mut ttft_s: Option<f64> = None;
    let mut total_s: Option<f64> = None;

    loop {
        let chunk = resp.chunk().await?;
        let Some(chunk) = chunk else { break };
        let s = std::str::from_utf8(&chunk)
            .map_err(|e| anyhow!("invalid utf-8 in SSE response: {e}"))?;
        buf.push_str(s);

        while let Some(idx) = buf.find("\n\n") {
            let event = buf[..idx].to_string();
            buf.drain(..idx + 2);

            for line in event.lines() {
                let Some(data) = line.strip_prefix("data:") else { continue };
                let data = data.trim();

                if data == "[DONE]" {
                    if print_deltas {
                        println!();
                    }
                    return Ok((out, ttft_s, total_s));
                }

                // Normal OpenAI streaming chunk.
                if data.starts_with('{') {
                    let v: serde_json::Value =
                        serde_json::from_str(data).map_err(|e| anyhow!("bad SSE json: {e}"))?;

                    if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
                        return Err(anyhow!("{err}"));
                    }

                    if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
                        if !c.is_empty() {
                            out.push_str(c);
                            if print_deltas {
                                print!("{c}");
                                std::io::stdout().flush().ok();
                            }
                        }
                    }

                    // Our server optionally attaches metrics on the final chunk.
                    if ttft_s.is_none() {
                        ttft_s = v.get("ttft_s").and_then(|x| x.as_f64());
                    }
                    if total_s.is_none() {
                        total_s = v.get("total_s").and_then(|x| x.as_f64());
                    }
                }
            }
        }
    }

    // Stream ended without a [DONE] marker (still return what we got).
    if print_deltas {
        println!();
    }
    Ok((out, ttft_s, total_s))
}

/// （新增）`spm-cli` 的“简易客户端模式”：
/// - 连接到已运行的 `--api` 服务端
/// - 自动拼 JSON、发送请求
/// - 只在终端打印 assistant 的纯文本内容（不输出整段 JSON）
///
/// 为什么要加：让第二个端口（API 服务）用起来更像“聊天”，而不是每次写 curl + JSON。
async fn run_api_client(args: Args) -> Result<()> {
    let base = args
        .api_client
        .as_deref()
        .map(normalize_api_base)
        .unwrap_or_default();
    if base.is_empty() {
        return Err(anyhow!("--api-client is empty"));
    }

    let url = format!("{base}/api/v1/chat/completions");
    let http = reqwest::Client::new();

    let mut messages: Vec<spm_core::models::chat::Message> = vec![];
    if !args.system_prompt.is_empty() {
        // 把 system_prompt 放到会话历史里，服务端按 OpenAI 兼容格式处理。
        messages.push(spm_core::models::chat::Message::system(
            args.system_prompt.clone(),
        ));
    }

    // Single-shot mode: --ask, else fallback to --prompt, else read stdin once.
    if !args.repl {
        let ask = args
            .ask
            .clone()
            .or_else(|| (!args.prompt.is_empty()).then(|| args.prompt.clone()));

        let ask = match ask {
            Some(s) => s,
            None => {
                let mut buf = String::new();
                BufReader::new(tokio::io::stdin())
                    .read_line(&mut buf)
                    .await?;
                buf.trim().to_string()
            }
        };

        if ask.is_empty() {
            return Err(anyhow!(
                "no prompt provided; use --ask, --prompt, or pipe text into stdin"
            ));
        }

        // 单次提问：支持纯文本或“文本+图片”（由 --image 控制）。
        messages.push(user_message_text_or_image(ask, args.image.as_deref())?);

        if args.stream {
            let resp = http
                .post(url)
                .json(&serde_json::json!({ "messages": messages, "stream": true }))
                .send()
                .await?
                .error_for_status()?;
            let (_content, ttft, total) = consume_sse_stream(resp, true).await?;
            if args.metrics {
                let ttft_s = ttft.map(|v| format!("{v:.3}")).unwrap_or_else(|| "null".to_string());
                let total_s = total.map(|v| format!("{v:.3}")).unwrap_or_else(|| "null".to_string());
                eprintln!("[metrics] ttft_s={} total_s={}", ttft_s, total_s);
            }
        } else {
            let resp: serde_json::Value = http
                .post(url)
                .json(&serde_json::json!({ "messages": messages }))
                .send()
                .await?
                .error_for_status()?
                .json()
                .await?;

            let content = resp["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .to_string();
            println!("{content}");

            // （新增）可选打印服务端返回的 TTFT/总耗时（单位秒）。写到 stderr，避免污染 stdout 的纯文本输出。
            if args.metrics {
                let ttft_s = resp["ttft_s"]
                    .as_f64()
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "null".to_string());
                let total_s = resp["total_s"]
                    .as_f64()
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "null".to_string());
                eprintln!("[metrics] ttft_s={} total_s={}", ttft_s, total_s);
            }
        }
        return Ok(());
    }

    // REPL mode.
    let mut stdin = BufReader::new(tokio::io::stdin());
    let mut line = String::new();

    if !args.prompt.is_empty() {
        // 可选：启动 REPL 前先发一条初始 prompt（也支持 --image）。
        messages.push(user_message_text_or_image(
            args.prompt.clone(),
            args.image.as_deref(),
        )?);
        if args.stream {
            let resp = http
                .post(&url)
                .json(&serde_json::json!({ "messages": messages, "stream": true }))
                .send()
                .await?
                .error_for_status()?;
            let (content, ttft, total) = consume_sse_stream(resp, true).await?;
            messages.push(spm_core::models::chat::Message::assistant(content));
            if args.metrics {
                eprintln!(
                    "[metrics] ttft_s={} total_s={}",
                    ttft.map(|v| format!("{v:.3}")).unwrap_or_else(|| "null".to_string()),
                    total.map(|v| format!("{v:.3}")).unwrap_or_else(|| "null".to_string())
                );
            }
        } else {
            let resp: serde_json::Value = http
                .post(&url)
                .json(&serde_json::json!({ "messages": messages }))
                .send()
                .await?
                .error_for_status()?
                .json()
                .await?;
            let content = resp["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .to_string();
            println!("{content}");
            messages.push(spm_core::models::chat::Message::assistant(content));

            if args.metrics {
                let ttft = resp["ttft_s"].as_f64();
                let total = resp["total_s"].as_f64();
                eprintln!(
                    "[metrics] ttft_s={} total_s={}",
                    ttft.map(|v| format!("{v:.3}")).unwrap_or_else(|| "null".to_string()),
                    total.map(|v| format!("{v:.3}")).unwrap_or_else(|| "null".to_string())
                );
            }
        }
    }

    loop {
        line.clear();
        // 交互提示符写到 stderr，方便把 stdout 重定向保存模型回复内容。
        eprint!("你> ");
        let n = stdin.read_line(&mut line).await?;
        if n == 0 {
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if matches!(input, "/q" | "/quit" | "/exit") {
            break;
        }

        // （新增）REPL 图片快捷指令：`/img path/to.png 你要问的问题`
        // 为什么要加：在 REPL 里临时换图片更方便，不需要退出重启进程/改 --image。
        let (user_text, img_path) = if let Some(rest) = input.strip_prefix("/img ") {
            let rest = rest.trim();
            let mut it = rest.splitn(2, char::is_whitespace);
            let path = it.next().unwrap_or("").trim();
            let question = it.next().unwrap_or("").trim();
            if path.is_empty() || question.is_empty() {
                println!("用法: /img <图片路径> <问题>");
                continue;
            }
            (question.to_string(), Some(path.to_string()))
        } else {
            (input.to_string(), None)
        };

        // 多模态优先级：如果本行用了 /img，就用它；否则复用启动时的 --image（如果有）。
        messages.push(user_message_text_or_image(
            user_text,
            img_path.as_deref().or(args.image.as_deref()),
        )?);
        if args.stream {
            let resp = http
                .post(&url)
                .json(&serde_json::json!({ "messages": messages, "stream": true }))
                .send()
                .await?
                .error_for_status()?;
            let (content, ttft, total) = consume_sse_stream(resp, true).await?;
            messages.push(spm_core::models::chat::Message::assistant(content));
            if args.metrics {
                eprintln!(
                    "[metrics] ttft_s={} total_s={}",
                    ttft.map(|v| format!("{v:.3}")).unwrap_or_else(|| "null".to_string()),
                    total.map(|v| format!("{v:.3}")).unwrap_or_else(|| "null".to_string())
                );
            }
        } else {
            let resp: serde_json::Value = http
                .post(&url)
                .json(&serde_json::json!({ "messages": messages }))
                .send()
                .await?
                .error_for_status()?
                .json()
                .await?;

            let content = resp["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .to_string();
            println!("{content}");
            messages.push(spm_core::models::chat::Message::assistant(content));

            if args.metrics {
                let ttft = resp["ttft_s"].as_f64();
                let total = resp["total_s"].as_f64();
                eprintln!(
                    "[metrics] first_token={} total_time={}",
                    ttft.map(|v| format!("{v:.3}")).unwrap_or_else(|| "null".to_string()),
                    total.map(|v| format!("{v:.3}")).unwrap_or_else(|| "null".to_string())
                );
            }
        }
    }

    Ok(())
}

fn detect_model_type(model_dir: &str) -> Result<SelectedModel> {
    let config_path = Path::new(model_dir).join("config.json");
    let raw =
        fs::read(&config_path).map_err(|e| anyhow!("can't read {}: {e}", config_path.display()))?;
    let v: serde_json::Value = serde_json::from_slice(&raw)
        .map_err(|e| anyhow!("can't parse {}: {e}", config_path.display()))?;
    Ok(match v.get("model_type") {
        Some(serde_json::Value::String(s)) if s == "qwen3_vl" => SelectedModel::Qwen3Vl,
        _ => SelectedModel::Llama3,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    // parse command line
    let args = Args::parse();

    // （新增）客户端模式：不加载本地模型、不启动 master/worker，只负责调用 API 并打印纯文本回复。
    // 为什么要加：你在“第二个端口（API）”上提问时，不想再手写 JSON。
    if args.api_client.is_some() {
        return run_api_client(args).await;
    }

    let selected_model = detect_model_type(&args.model)?;

    // setup logging
    if std::env::var_os("RUST_LOG").is_none() {
        // set `RUST_LOG=debug` to see debug logs
        std::env::set_var("RUST_LOG", "info,tokenizers=error,actix_server=warn");
    }

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_module_path(false)
        .format_target(false)
        .init();

    // setup context
    let ctx = Context::from_args(args)?;
    log::info!("selected model: {:?}", selected_model);

    // run either in master or worker mode depending on command line
    let ret = match (ctx.args.mode.clone(), selected_model) {
        (Mode::Master, SelectedModel::Qwen3Vl) => {
            Master::<spm_core::models::qwen3_vl::Qwen3Vl>::new(ctx)
                .await?
                .run()
                .await
        }
        (Mode::Worker, SelectedModel::Qwen3Vl) => {
            Worker::<spm_core::models::qwen3_vl::Qwen3Vl>::new(ctx)
                .await?
                .run()
                .await
        }
        (Mode::Master, SelectedModel::Llama3) => {
            Master::<spm_core::models::llama3::LLama>::new(ctx)
                .await?
                .run()
                .await
        }
        (Mode::Worker, SelectedModel::Llama3) => {
            Worker::<spm_core::models::llama3::LLama>::new(ctx)
                .await?
                .run()
                .await
        }
    };

    if ret.is_err() {
        // we were possibly streaming text, add a newline before reporting the error
        println!();
        return ret;
    }

    Ok(())
}
