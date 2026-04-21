mod constants;
mod dedup;
mod domains;
mod generation;
mod models;
mod prompts;
mod readme;

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

use crate::constants::LANGUAGES;
use crate::domains::DOMAINS;
use crate::generation::stats::Stats;
use crate::models::{RunStats, TaskEntry};
use crate::prompts::build_system_prompt;
use crate::readme::generate_readme;

#[derive(Debug, Parser)]
#[command(name = "taskgen")]
struct Args {
    #[arg(long, default_value = "https://api.openai.com/v1")]
    api_base: String,

    #[arg(long, env = "OPENAI_API_KEY")]
    api_key: Option<String>,

    #[arg(short, long, default_value = "gpt-4o-mini")]
    model: String,

    #[arg(long)]
    keyfile: Option<String>,

    #[arg(long)]
    input_price: Option<f64>,
    #[arg(long)]
    output_price: Option<f64>,

    #[arg(short, long, default_value_t = 0.9)]
    temperature: f64,

    #[arg(short = 'c', long, default_value_t = 250)]
    count: usize,

    #[arg(short, long, default_value = "output.jsonl")]
    output: PathBuf,

    #[arg(long)]
    system_prompt: Option<String>,

    #[arg(long)]
    distribution: Option<String>,

    #[arg(long)]
    difficulty: Option<String>,

    #[arg(long)]
    seed: Option<u64>,

    #[arg(long)]
    dedup: bool,

    #[arg(long, default_value_t = 0.6)]
    dedup_threshold: f64,

    #[arg(long)]
    free_models: bool,

    /// Rescan interval in minutes for free model availability (default: 10)
    #[arg(long, default_value_t = 10)]
    free_rescan: u64,

    #[arg(long)]
    budget: Option<f64>,

    #[arg(short = 'w', long, default_value_t = 5)]
    workers: usize,

    #[arg(long)]
    append: bool,

    #[arg(long)]
    proxies: Option<PathBuf>,

    #[arg(long)]
    rotating_proxy: bool,

    #[arg(long)]
    lang: Option<String>,

    #[arg(long)]
    multilingual: bool,
}

// ── API types ────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f64,
    max_tokens: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    usage: Option<Usage>,
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ChatMessage,
}

// ── Error types ──────────────────────────────────────────────────────────────

enum ApiError {
    RateLimit(Option<u64>),
    Billing(String),
    Timeout,
    Other(anyhow::Error),
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::RateLimit(s) => write!(f, "rate limited (retry after {:?}s)", s),
            ApiError::Billing(msg) => write!(f, "billing error: {}", msg),
            ApiError::Timeout => write!(f, "request timed out"),
            ApiError::Other(e) => write!(f, "{}", e),
        }
    }
}

fn is_billing_error(status: reqwest::StatusCode, body: &str) -> bool {
    if status.as_u16() == 402 {
        return true;
    }
    let lower = body.to_lowercase();
    lower.contains("insufficient_quota")
        || lower.contains("billing")
        || lower.contains("payment required")
        || lower.contains("exceeded your current quota")
        || lower.contains("account is not active")
        || lower.contains("insufficient_funds")
        || lower.contains("budget")
}

// ── Free-model discovery (OpenRouter) ────────────────────────────────────────

const OPENROUTER_API_BASE: &str = "https://openrouter.ai/api/v1";
const MIN_FREE_MODEL_CTX: u64 = 16000;

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
    name: String,
    architecture: ModelArchitecture,
    pricing: ModelPricing,
    top_provider: ModelProvider,
}

#[derive(Debug, Deserialize)]
struct ModelArchitecture {
    input_modalities: Vec<String>,
    output_modalities: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ModelPricing {
    prompt: String,
    completion: String,
}

#[derive(Debug, Deserialize)]
struct ModelProvider {
    context_length: Option<u64>,
}

async fn fetch_free_models(client: &reqwest::Client, api_key: &str) -> anyhow::Result<Vec<String>> {
    let url = format!("{}/models", OPENROUTER_API_BASE);
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .send()
        .await
        .context("failed to fetch OpenRouter models")?;

    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("OpenRouter models API error: {}", text);
    }

    let models: ModelsResponse = resp.json().await.context("failed to parse models response")?;

    let mut free: Vec<(String, String, u64)> = models
        .data
        .into_iter()
        .filter(|m| {
            m.pricing.prompt == "0"
                && m.pricing.completion == "0"
                && m.architecture.input_modalities.contains(&"text".to_string())
                && m.architecture.output_modalities.contains(&"text".to_string())
                && m.id != "openrouter/free"
                && m.top_provider.context_length.unwrap_or(0) >= MIN_FREE_MODEL_CTX
        })
        .map(|m| {
            let ctx = m.top_provider.context_length.unwrap_or(0);
            (m.id, m.name, ctx)
        })
        .collect();

    free.sort_by(|a, b| b.2.cmp(&a.2));

    if free.is_empty() {
        anyhow::bail!(
            "no free models with >= {}k context available on OpenRouter right now",
            MIN_FREE_MODEL_CTX / 1000
        );
    }

    println!("Found {} candidate free models, running health checks...", free.len());

    let mut verified: Vec<String> = Vec::new();
    for (id, name, ctx) in &free {
        print!("  testing {} ({}, {}k ctx)... ", id, name, ctx / 1000);
        match test_model(client, api_key, id).await {
            Ok(()) => {
                println!("ok");
                verified.push(id.clone());
            }
            Err(e) => {
                println!("offline ({})", e);
            }
        }
    }

    if verified.is_empty() {
        anyhow::bail!("all free models are offline on OpenRouter right now");
    }

    println!("Using {} verified free models", verified.len());
    Ok(verified)
}

async fn test_model(client: &reqwest::Client, api_key: &str, model: &str) -> anyhow::Result<()> {
    let body = ChatRequest {
        model: model.to_string(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "Say hi.".into(),
        }],
        temperature: 0.0,
        max_tokens: Some(5),
    };

    let url = format!("{}/chat/completions", OPENROUTER_API_BASE);
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .timeout(std::time::Duration::from_secs(15))
        .send()
        .await
        .context("request failed")?;

    let status = resp.status();
    if status.as_u16() == 429 {
        return Ok(());
    }
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("{}: {}", status, &text[..text.len().min(100)]);
    }
    let chat_resp: ChatResponse = resp.json().await.context("bad response")?;
    if chat_resp.choices.is_empty() {
        anyhow::bail!("no choices returned");
    }
    Ok(())
}

// ── Per-model failure tracking ────────────────────────────────────────────────

const MAX_MODEL_FAILURES: usize = 3;

struct ModelFailures {
    counts: std::sync::Mutex<HashMap<String, usize>>,
    rescan_notify: tokio::sync::Notify,
}

impl ModelFailures {
    fn new() -> Self {
        Self {
            counts: std::sync::Mutex::new(HashMap::new()),
            rescan_notify: tokio::sync::Notify::new(),
        }
    }

    fn record(&self, model: &str) -> bool {
        let mut counts = self.counts.lock().unwrap();
        let count = counts.entry(model.to_string()).or_insert(0);
        *count += 1;
        *count == MAX_MODEL_FAILURES
    }

    fn reset(&self) {
        self.counts.lock().unwrap().clear();
    }
}

// ── Proxy helpers ─────────────────────────────────────────────────────────────

fn parse_proxy_line(line: &str) -> anyhow::Result<reqwest::Proxy> {
    let line = line.trim();
    let parts: Vec<&str> = line.split(':').collect();
    let proxy_url = match parts.len() {
        2 => format!("http://{}:{}", parts[0], parts[1]),
        4 => format!("http://{}:{}@{}:{}", parts[2], parts[3], parts[0], parts[1]),
        _ => anyhow::bail!(
            "invalid proxy format '{}', expected host:port or host:port:user:pass",
            line
        ),
    };
    reqwest::Proxy::all(&proxy_url)
        .context(format!("failed to create proxy from '{}'", line))
}

fn load_proxies(path: &PathBuf) -> anyhow::Result<Vec<reqwest::Proxy>> {
    use std::io::BufRead;
    let file = File::open(path).context(format!("failed to open proxy file: {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut proxies = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.context("failed to read proxy file")?;
        let line = line.trim().to_string();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        proxies.push(parse_proxy_line(&line).context(format!("proxy line {}", i + 1))?);
    }
    if proxies.is_empty() {
        anyhow::bail!("proxy file is empty: {}", path.display());
    }
    Ok(proxies)
}

fn build_clients(proxies: &[reqwest::Proxy]) -> Vec<reqwest::Client> {
    proxies
        .iter()
        .map(|p| {
            reqwest::Client::builder()
                .proxy(p.clone())
                .build()
                .expect("failed to build client with proxy")
        })
        .collect()
}

// ── Core API call ─────────────────────────────────────────────────────────────

const MAX_RETRIES: u32 = 5;

async fn api_request(
    client: &reqwest::Client,
    url: &str,
    api_key: &str,
    body: &ChatRequest,
) -> std::result::Result<(String, u64, u64), ApiError> {
    let resp = client
        .post(url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(body)
        .send()
        .await;

    let resp = match resp {
        Ok(r) => r,
        Err(e) => {
            if e.is_timeout() {
                return Err(ApiError::Timeout);
            }
            return Err(ApiError::Other(e.into()));
        }
    };

    let status = resp.status();

    if status.as_u16() == 429 {
        let retry_after = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok());
        return Err(ApiError::RateLimit(retry_after));
    }

    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        if is_billing_error(status, &text) {
            return Err(ApiError::Billing(text));
        }
        return Err(ApiError::Other(anyhow::anyhow!("API error {}: {}", status, text)));
    }

    let chat_resp: ChatResponse = resp
        .json()
        .await
        .map_err(|e| ApiError::Other(anyhow::anyhow!("failed to parse API response: {}", e)))?;
    let choice = chat_resp
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| ApiError::Other(anyhow::anyhow!("no choices in response")))?;
    let prompt_text = choice.message.content.trim().to_string();

    let (input_tokens, output_tokens) = match chat_resp.usage {
        Some(u) => (u.prompt_tokens, u.completion_tokens),
        None => (0, 0),
    };

    Ok((prompt_text, input_tokens, output_tokens))
}

async fn generate_task(
    client: &reqwest::Client,
    api_base: &str,
    api_key: &str,
    model: &str,
    system_prompt: &str,
    domain_display: &str,
    subdomain: &str,
    difficulty: u8,
    temperature: f64,
    language: Option<&str>,
    cancel: &AtomicBool,
    consecutive_timeouts: &AtomicUsize,
    pb: &ProgressBar,
) -> std::result::Result<(String, u64, u64), ApiError> {
    let lang_instruction = match language {
        Some(code) if code != "en" => {
            let lang_name = LANGUAGES
                .iter()
                .find(|(c, _)| *c == code)
                .map(|(_, n)| *n)
                .unwrap_or("English");
            format!(
                "\n\nIMPORTANT: Write the entire task/prompt in {}. Do NOT use English.",
                lang_name
            )
        }
        _ => String::new(),
    };

    let user_msg = format!(
        "Generate a task/prompt for the following:\n\nDomain: {}\nSubdomain: {}\nDifficulty: {}/10\n\nOutput only the task prompt, nothing else.{}",
        domain_display, subdomain, difficulty, lang_instruction
    );

    let body = ChatRequest {
        model: model.to_string(),
        messages: vec![
            ChatMessage { role: "system".into(), content: system_prompt.into() },
            ChatMessage { role: "user".into(), content: user_msg },
        ],
        temperature,
        max_tokens: Some(2048),
    };

    let url = format!("{}/chat/completions", api_base.trim_end_matches('/'));

    let mut retries = 0u32;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(ApiError::Other(anyhow::anyhow!("cancelled")));
        }

        match api_request(client, &url, api_key, &body).await {
            Ok(result) => {
                consecutive_timeouts.store(0, Ordering::Relaxed);
                return Ok(result);
            }
            Err(ApiError::RateLimit(retry_after)) => {
                retries += 1;
                if retries > MAX_RETRIES {
                    return Err(ApiError::RateLimit(retry_after));
                }
                let wait = retry_after.unwrap_or_else(|| 2u64.pow(retries).min(60));
                pb.suspend(|| {
                    eprintln!("[RATE] 429 hit, waiting {}s (retry {}/{})", wait, retries, MAX_RETRIES);
                });
                tokio::time::sleep(tokio::time::Duration::from_secs(wait)).await;
            }
            Err(ApiError::Timeout) => {
                let count = consecutive_timeouts.fetch_add(1, Ordering::Relaxed) + 1;
                if count >= 5 {
                    pb.suspend(|| {
                        eprintln!("[FATAL] {} consecutive timeouts, shutting down gracefully...", count);
                    });
                    cancel.store(true, Ordering::Relaxed);
                    return Err(ApiError::Timeout);
                }
                retries += 1;
                if retries > MAX_RETRIES {
                    return Err(ApiError::Timeout);
                }
                let wait = 2u64.pow(retries).min(30);
                pb.suspend(|| {
                    eprintln!(
                        "[TIMEOUT] request timed out, waiting {}s (retry {}/{}, {} consecutive)",
                        wait, retries, MAX_RETRIES, count
                    );
                });
                tokio::time::sleep(tokio::time::Duration::from_secs(wait)).await;
            }
            Err(ApiError::Billing(msg)) => {
                pb.suspend(|| {
                    eprintln!("[FATAL] billing error, shutting down gracefully: {}", msg);
                });
                cancel.store(true, Ordering::Relaxed);
                return Err(ApiError::Billing(msg));
            }
            Err(e) => return Err(e),
        }
    }
}

// ── Misc helpers ──────────────────────────────────────────────────────────────

fn parse_distribution(s: &str) -> HashMap<String, f64> {
    let mut map = HashMap::new();
    for part in s.split(',') {
        let kv: Vec<&str> = part.trim().split(':').collect();
        if kv.len() == 2 {
            let domain = kv[0].trim();
            let weight: f64 = kv[1].trim().parse().unwrap_or(1.0);
            map.insert(domain.to_string(), weight);
        }
    }
    if map.is_empty() {
        for d in DOMAINS {
            map.insert(d.name.to_string(), 1.0);
        }
    }
    map
}

fn parse_difficulty(s: &str) -> HashMap<u8, f64> {
    let mut map = HashMap::new();
    for part in s.split(',') {
        let kv: Vec<&str> = part.trim().split(':').collect();
        if kv.len() == 2 {
            let level: u8 = kv[0].trim().parse().unwrap_or(5);
            let weight: f64 = kv[1].trim().parse().unwrap_or(1.0);
            map.insert(level, weight);
        }
    }
    map
}

fn sample_subdomains(rng: &mut impl Rng) -> (String, String, String) {
    let domain = &DOMAINS[rng.gen_range(0..DOMAINS.len())];
    let cat = domain.category.to_string();
    let name = domain.name.to_string();
    let sub = domain.subdomains[rng.gen_range(0..domain.subdomains.len())].to_string();
    (cat, name, sub)
}

fn count_existing_tasks(path: &PathBuf) -> usize {
    match File::open(path) {
        Ok(f) => BufReader::new(f).lines().filter_map(|l| l.ok()).filter(|l| !l.trim().is_empty()).count(),
        Err(_) => 0,
    }
}

fn word_trigrams(text: &str) -> HashSet<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut trigrams = HashSet::new();
    for window in words.windows(3) {
        trigrams.insert(window.join(" "));
    }
    trigrams
}

fn jaccard_similarity(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    intersection as f64 / union as f64
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let api_keys: Arc<Vec<String>> = Arc::new(match &args.keyfile {
        Some(kf) => {
            let content = std::fs::read_to_string(kf)?;
            content.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect()
        }
        None => {
            let key = args.api_key.clone().context("API key required. Use --api-key, set OPENAI_API_KEY, or use --keyfile")?;
            vec![key]
        }
    });
    let key_counter = Arc::new(AtomicUsize::new(0));

    let api_base = if args.free_models {
        OPENROUTER_API_BASE.to_string()
    } else {
        args.api_base.clone()
    };

    let model_failures = Arc::new(ModelFailures::new());

    let free_model_list: Option<Arc<tokio::sync::RwLock<Vec<String>>>> = if args.free_models {
        let discovery_client = reqwest::Client::new();
        let models = fetch_free_models(&discovery_client, &api_keys[0]).await?;
        Some(Arc::new(tokio::sync::RwLock::new(models)))
    } else {
        None
    };
    let model_counter = Arc::new(AtomicUsize::new(0));

    let dist = match &args.distribution {
        Some(d) => parse_distribution(d),
        None => {
            let mut m = HashMap::new();
            for domain in DOMAINS {
                m.insert(domain.name.to_string(), 1.0);
            }
            m
        }
    };

    let diff_dist = match &args.difficulty {
        Some(d) => parse_difficulty(d),
        None => {
            let mut m = HashMap::new();
            for i in 1..=10 {
                m.insert(i, 1.0);
            }
            m
        }
    };

    let custom_prompt = args.system_prompt.clone();
    let existing = if args.append { count_existing_tasks(&args.output) } else { 0 };
    if existing > 0 {
        println!("Appending to existing file with {} tasks", existing);
    }

    let file = if args.append && args.output.exists() {
        OpenOptions::new().append(true).open(&args.output)?
    } else {
        File::create(&args.output)?
    };

    // Build HTTP clients (with optional proxy support)
    let clients: Arc<Vec<reqwest::Client>> = Arc::new(match &args.proxies {
        Some(proxy_path) => {
            let proxies = load_proxies(proxy_path)?;
            let total = proxies.len();
            if args.rotating_proxy {
                let idx = thread_rng().gen_range(0..total);
                println!("Using rotating proxy (sticky): proxy #{}", idx + 1);
                vec![reqwest::Client::builder()
                    .proxy(proxies.into_iter().nth(idx).unwrap())
                    .build()?]
            } else {
                println!("Loaded {} proxies (round-robin)", total);
                build_clients(&proxies)
            }
        }
        None => vec![reqwest::Client::new()],
    });
    let proxy_counter = Arc::new(AtomicUsize::new(0));

    let stats = Arc::new(Stats::new());
    let file = Arc::new(std::sync::Mutex::new(file));
    let cancel = Arc::new(AtomicBool::new(false));
    let consecutive_timeouts = Arc::new(AtomicUsize::new(0));

    let budget = args.budget;
    let input_price = args.input_price;
    let output_price = args.output_price;
    let count = args.count;
    let workers = args.workers;

    // Pre-sample all (domain, difficulty, language) tuples up front to avoid
    // RNG contention inside the concurrent workers.
    let mut rng = thread_rng();
    let multilingual = args.multilingual;
    let presampled: Vec<(String, String, String, u8, Option<String>)> = (0..count)
        .map(|_| {
            let (cat, name, sub) = sample_subdomains(&mut rng);
            let diff = loop {
                let level: u8 = rng.gen_range(1..=10);
                if diff_dist.contains_key(&level) {
                    break level;
                }
            };
            let lang = if multilingual {
                let idx = rng.gen_range(0..LANGUAGES.len());
                Some(LANGUAGES[idx].0.to_string())
            } else {
                args.lang.clone()
            };
            (cat, name, sub, diff, lang)
        })
        .collect();
    let presampled = Arc::new(presampled);

    let pb = ProgressBar::new(count as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}) | {msg}")
            .unwrap()
            .progress_chars("##-"),
    );
    pb.set_message("starting...");

    // Background task: periodically refresh the free-model list.
    let rescan_handle = if let Some(ref model_list) = free_model_list {
        let model_list = model_list.clone();
        let cancel = cancel.clone();
        let model_failures = model_failures.clone();
        let api_key = api_keys[0].clone();
        let rescan_mins = args.free_rescan;
        let pb = pb.clone();
        Some(tokio::spawn(async move {
            let client = reqwest::Client::new();
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(tokio::time::Duration::from_secs(rescan_mins * 60)) => {},
                    _ = model_failures.rescan_notify.notified() => {},
                }
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                pb.suspend(|| println!("[RESCAN] refreshing free model list..."));
                match fetch_free_models(&client, &api_key).await {
                    Ok(new_models) => {
                        let n = new_models.len();
                        model_failures.reset();
                        *model_list.write().await = new_models;
                        pb.suspend(|| println!("[RESCAN] updated: {} models available", n));
                    }
                    Err(e) => {
                        pb.suspend(|| eprintln!("[RESCAN] failed to refresh: {}, keeping current list", e));
                    }
                }
            }
        }))
    } else {
        None
    };

    stream::iter(0..count).for_each_concurrent(workers, |i| {
        let clients = clients.clone();
        let proxy_counter = proxy_counter.clone();
        let api_keys = api_keys.clone();
        let key_counter = key_counter.clone();
        let api_base = api_base.clone();
        let stats = stats.clone();
        let cancel = cancel.clone();
        let consecutive_timeouts = consecutive_timeouts.clone();
        let free_model_list = free_model_list.clone();
        let model_counter = model_counter.clone();
        let model_failures = model_failures.clone();
        let pb = pb.clone();
        let custom_prompt = custom_prompt.clone();
        let model = args.model.clone();
        let temperature = args.temperature;
        let file = file.clone();
        let presampled = presampled.clone();

        async move {
            if cancel.load(Ordering::Relaxed) {
                pb.inc(1);
                return;
            }

            let (ref cat, ref domain_name, ref subdomain, difficulty, ref lang) = presampled[i];

            // Budget guard
            if let (Some(b), Some(ip), Some(op)) = (budget, input_price, output_price) {
                let in_tok = stats.input_tokens.load(Ordering::Relaxed) as f64;
                let out_tok = stats.output_tokens.load(Ordering::Relaxed) as f64;
                if (ip * in_tok / 1_000_000.0) + (op * out_tok / 1_000_000.0) >= b {
                    cancel.store(true, Ordering::Relaxed);
                    pb.inc(1);
                    return;
                }
            }

            let use_model = match &free_model_list {
                Some(models) => {
                    let list = models.read().await;
                    let idx = model_counter.fetch_add(1, Ordering::Relaxed) % list.len();
                    list[idx].clone()
                }
                None => model.clone(),
            };

            let domain_display = format!("{}::{}", cat, domain_name);
            let client_idx = proxy_counter.fetch_add(1, Ordering::Relaxed) % clients.len();
            let client = &clients[client_idx];
            let key_idx = key_counter.fetch_add(1, Ordering::Relaxed) % api_keys.len();
            let api_key = &api_keys[key_idx];

            let system_prompt: String = match custom_prompt.as_ref() {
                Some(p) => p.clone(),
                None => build_system_prompt(lang.as_deref()),
            };

            match generate_task(
                client,
                &api_base,
                api_key,
                &use_model,
                &system_prompt,
                &domain_display,
                subdomain,
                difficulty,
                temperature,
                lang.as_deref(),
                &cancel,
                &consecutive_timeouts,
                &pb,
            )
            .await
            {
                Ok((prompt_text, in_tok, out_tok)) => {
                    if prompt_text.trim().is_empty() {
                        stats.errors.fetch_add(1, Ordering::Relaxed);
                        pb.inc(1);
                        return;
                    }
                    let entry = TaskEntry {
                        prompt: prompt_text,
                        domain: domain_display,
                        subdomain: subdomain.clone(),
                        difficulty,
                        language: lang.clone(),
                    };
                    let line = serde_json::to_string(&entry).unwrap() + "\n";
                    {
                        let mut f = file.lock().unwrap();
                        let _ = f.write_all(line.as_bytes());
                        let _ = f.flush();
                    }
                    stats.input_tokens.fetch_add(in_tok, Ordering::Relaxed);
                    stats.output_tokens.fetch_add(out_tok, Ordering::Relaxed);
                    let done = stats.tasks.fetch_add(1, Ordering::Relaxed) + 1;
                    let errs = stats.errors.load(Ordering::Relaxed);
                    let cur_in = stats.input_tokens.load(Ordering::Relaxed) as f64;
                    let cur_out = stats.output_tokens.load(Ordering::Relaxed) as f64;
                    let total_tok = (cur_in + cur_out) as u64;
                    let cost_str = match (input_price, output_price) {
                        (Some(ip), Some(op)) => {
                            let cost = (ip * cur_in / 1_000_000.0) + (op * cur_out / 1_000_000.0);
                            if let Some(b) = budget {
                                if cost >= b {
                                    cancel.store(true, Ordering::Relaxed);
                                }
                            }
                            format!(" | ${:.4}", cost)
                        }
                        _ => String::new(),
                    };
                    pb.set_message(format!(
                        "{} ok | {} err | {}k tok{}",
                        done, errs, total_tok / 1000, cost_str
                    ));
                }
                Err(e) => {
                    stats.errors.fetch_add(1, Ordering::Relaxed);
                    if !cancel.load(Ordering::Relaxed) {
                        pb.suspend(|| eprintln!("[ERROR] task {}: {}", i + 1, e));
                    }
                    if free_model_list.is_some() {
                        let tripped = model_failures.record(&use_model);
                        if tripped {
                            pb.suspend(|| {
                                eprintln!(
                                    "[RESCAN] {} failed {} times, marking offline and triggering rescan",
                                    use_model, MAX_MODEL_FAILURES
                                );
                            });
                            model_failures.rescan_notify.notify_one();
                        }
                    }
                }
            }
            pb.inc(1);
        }
    }).await;

    if let Some(handle) = rescan_handle {
        handle.abort();
    }

    let was_cancelled = cancel.load(Ordering::Relaxed);
    if was_cancelled {
        pb.finish_with_message("stopped early — saving progress");
    } else {
        pb.finish_with_message("done");
    }

    let total_errors = stats.errors.load(Ordering::Relaxed);
    let total_in = stats.input_tokens.load(Ordering::Relaxed);
    let total_out = stats.output_tokens.load(Ordering::Relaxed);
    let total_tasks = stats.tasks.load(Ordering::Relaxed);

    if was_cancelled {
        println!("\nGraceful shutdown — saved {} tasks before exit", total_tasks);
    }
    println!("Generated {} tasks ({} errors)", total_tasks, total_errors);
    println!("Tokens: {} in / {} out", total_in, total_out);

    println!("Generated {} tasks ({} errors)", total_tasks, total_errors);
    println!("Tokens: {} in / {} out", total_in, total_out);

    let run_stats = RunStats {
        total_input_tokens: total_in,
        total_output_tokens: total_out,
        total_tasks,
        errors: total_errors,
    };

    if args.dedup && args.output.exists() {
        println!("\nRunning deduplication (threshold: {:.2})...", args.dedup_threshold);

        let reader = BufReader::new(File::open(&args.output)?);
        let mut lines: Vec<String> = Vec::new();
        let mut entries: Vec<Option<TaskEntry>> = Vec::new();

        for line in reader.lines().flatten() {
            let entry = serde_json::from_str::<TaskEntry>(&line).ok();
            entries.push(entry);
            lines.push(line);
        }

        let mut seen: HashSet<String> = HashSet::new();
        let mut keep = vec![true; lines.len()];
        let mut exact_dupes = 0usize;

        for (i, entry) in entries.iter().enumerate() {
            if let Some(e) = entry {
                let normalized: String = e.prompt.to_lowercase().split_whitespace().collect();
                if !seen.insert(normalized) {
                    keep[i] = false;
                    exact_dupes += 1;
                }
            }
        }

        if exact_dupes > 0 {
            println!("Removed {} exact duplicates", exact_dupes);
        }

        let kept_indices: Vec<usize> = (0..lines.len()).filter(|&i| keep[i]).collect();
        let trigrams: Vec<Option<HashSet<String>>> = kept_indices
            .iter()
            .map(|&i| {
                entries[i]
                    .as_ref()
                    .map(|e| word_trigrams(&e.prompt.to_lowercase()))
            })
            .collect();

        let mut semantic_dupes = 0usize;
        for j in 1..kept_indices.len() {
            if !keep[kept_indices[j]] {
                continue;
            }
            let trig_b = match &trigrams[j] {
                Some(t) => t,
                None => continue,
            };
            for k in 0..j {
                if !keep[kept_indices[k]] {
                    continue;
                }
                let trig_a = match &trigrams[k] {
                    Some(t) => t,
                    None => continue,
                };
                if jaccard_similarity(trig_a, trig_b) >= args.dedup_threshold {
                    keep[kept_indices[j]] = false;
                    semantic_dupes += 1;
                    break;
                }
            }
        }

        if semantic_dupes > 0 {
            println!("Removed {} semantic duplicates (similarity >= {:.2})", semantic_dupes, args.dedup_threshold);
        }

        let total_removed = exact_dupes + semantic_dupes;
        if total_removed > 0 {
            let mut f = File::create(&args.output)?;
            for (i, line) in lines.iter().enumerate() {
                if keep[i] {
                    f.write_all(line.as_bytes())?;
                    f.write_all(b"\n")?;
                }
            }
            let remaining = lines.len() - total_removed;
            println!("Deduplication complete: {} removed, {} remaining", total_removed, remaining);
        } else {
            println!("No duplicates found");
        }
    }

    let lang_counts: Option<HashMap<String, usize>> = if args.multilingual && args.output.exists() {
        println!("\nSplitting output by language...");
        let reader = BufReader::new(File::open(&args.output)?);
        let mut lang_buckets: HashMap<String, Vec<String>> = HashMap::new();

        for line in reader.lines().flatten() {
            let lang_code = serde_json::from_str::<serde_json::Value>(&line)
                .ok()
                .and_then(|v| v.get("language").and_then(|l| l.as_str().map(|s| s.to_string())))
                .unwrap_or_else(|| "en".to_string());
            lang_buckets.entry(lang_code).or_default().push(line);
        }

        let out_dir = args.output.parent().unwrap_or(std::path::Path::new("."));
        let stem = args.output.file_stem().unwrap_or_default().to_string_lossy();
        let ext = args.output.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();

        let counts: HashMap<String, usize> = lang_buckets.iter().map(|(k, v)| (k.clone(), v.len())).collect();

        for (lang, lines) in &lang_buckets {
            let lang_path = out_dir.join(format!("{}_{}{}", stem, lang, ext));
            let mut f = File::create(&lang_path)?;
            for line in lines {
                f.write_all(line.as_bytes())?;
                f.write_all(b"\n")?;
            }
            println!("  {} — {} tasks -> {}", lang, lines.len(), lang_path.display());
        }

        Some(counts)
    } else {
        None
    };

    let readme = generate_readme(&args, &run_stats, &dist, &diff_dist, lang_counts.as_ref());
    let readme_path = args.output.parent().unwrap_or(std::path::Path::new(".")).join("README.md");
    let mut rf = File::create(&readme_path).context("failed to create README.md")?;
    rf.write_all(readme.as_bytes())?;
    println!("README.md written to {}", readme_path.display());

    Ok(())
}
