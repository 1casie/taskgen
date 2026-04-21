
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

pub fn get_all_prompts() -> Vec<(&'static str, &'static str)> {
    vec![
        ("en", "English"),
        ("de", "German"),
        ("fr", "French"),
        ("es", "Spanish"),
        ("nl", "Dutch"),
        ("zh", "Chinese"),
        ("ar", "Arabic"),
        ("ru", "Russian"),
    ]
}
