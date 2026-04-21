use crate::constants::LANGUAGES;

pub mod stats {
    use std::sync::atomic::{AtomicU64, AtomicUsize};

    pub struct Stats {
        pub tasks: AtomicUsize,
        pub errors: AtomicUsize,
        pub input_tokens: AtomicU64,
        pub output_tokens: AtomicU64,
    }

    impl Stats {
        pub fn new() -> Self {
            Self {
                tasks: AtomicUsize::new(0),
                errors: AtomicUsize::new(0),
                input_tokens: AtomicU64::new(0),
                output_tokens: AtomicU64::new(0),
            }
        }
    }
}

pub struct LanguageSelector {
    pub languages: Vec<String>,
    index: std::sync::atomic::AtomicUsize,
}

impl LanguageSelector {
    pub fn new(codes: Vec<String>) -> Self {
        Self {
            languages: codes,
            index: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn next(&self) -> Option<String> {
        let idx = self.index.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % self.languages.len();
        Some(self.languages[idx].clone())
    }
}

pub fn parse_user_languages(lang_str: &str) -> Vec<String> {
    lang_str
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

pub fn validate_languages(langs: &[String]) -> Vec<String> {
    let valid_codes: Vec<&str> = LANGUAGES.iter().map(|(c, _)| *c).collect();
    langs
        .iter()
        .filter(|l| valid_codes.contains(&l.as_str()))
        .cloned()
        .collect()
}

pub fn build_system_prompt(language: Option<&str>) -> String {
    match language {
        Some("de") => include_str!("prompts/de.txt").to_string(),
        Some("fr") => include_str!("prompts/fr.txt").to_string(),
        Some("es") => include_str!("prompts/es.txt").to_string(),
        Some("nl") => include_str!("prompts/nl.txt").to_string(),
        Some("ar") => include_str!("prompts/ar.txt").to_string(),
        Some("ru") => include_str!("prompts/ru.txt").to_string(),
        Some("zh") => include_str!("prompts/zh.txt").to_string(),
        _ => include_str!("prompts/en.txt").to_string(),
    }
}
