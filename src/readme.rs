use std::collections::HashMap;

use crate::constants::LANGUAGES;
use crate::models::RunStats;

pub fn generate_readme(
    args: &crate::Args,
    stats: &RunStats,
    dist: &HashMap<String, f64>,
    diff_dist: &HashMap<u8, f64>,
    lang_counts: Option<&HashMap<String, usize>>,
) -> String {
    let mut md = String::new();
    md.push_str("# Task Generation Summary\n\n");
    md.push_str("## Configuration\n\n");
    md.push_str(&format!("| Setting | Value |\n|---|---|\n"));
    md.push_str(&format!("| Tasks | {} |\n", args.count));
    md.push_str(&format!("| Model | `{}` |\n", args.model));
    md.push_str(&format!("| Temperature | `{}` |\n", args.temperature));
    md.push_str(&format!("| Concurrency | {} workers |\n", args.workers));
    md.push_str(&format!("| API Base | `{}` |\n", args.api_base));

    if let Some(b) = args.budget {
        md.push_str(&format!("| Budget | ${:.2} |\n", b));
    }

    md.push_str(&format!("| Dedup | {} |\n", if args.dedup { "yes" } else { "no" }));

    md.push_str("\n## Statistics\n\n");
    md.push_str(&format!("- Tasks: {}\n", stats.total_tasks));
    md.push_str(&format!("- Errors: {}\n", stats.errors));
    md.push_str(&format!("- Input tokens: {}\n", stats.total_input_tokens));
    md.push_str(&format!("- Output tokens: {}\n", stats.total_output_tokens));

    if let Some(counts) = lang_counts {
        md.push_str("\n## Language Distribution\n\n");
        md.push_str("| Language | Code | Tasks |\n|---|---|---|\n");
        for (code, count) in counts {
            let name = LANGUAGES.iter()
                .find(|(c, _)| *c == code.as_str())
                .map(|(_, n)| *n)
                .unwrap_or("Unknown");
            md.push_str(&format!("| {} | `{}` | {} |\n", name, code, count));
        }
    }

    md.push_str("\n## Domain Distribution\n\n");
    md.push_str("| Domain | Weight |\n|---|---|\n");
    for (domain, weight) in dist {
        md.push_str(&format!("| {} | {:.2} |\n", domain, weight));
    }

    md.push_str("\n## Difficulty Distribution\n\n");
    md.push_str("| Level | Weight |\n|---|---|\n");
    for (level, weight) in diff_dist {
        md.push_str(&format!("| {} | {:.2} |\n", level, weight));
    }

    md
}
