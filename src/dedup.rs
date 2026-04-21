use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

use crate::models::TaskEntry;

pub fn word_trigrams(text: &str) -> HashSet<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut trigrams = HashSet::new();
    for window in words.windows(3) {
        trigrams.insert(window.join(" "));
    }
    trigrams
}

pub fn jaccard_similarity(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    intersection as f64 / union as f64
}

pub fn run_deduplication(
    path: &std::path::Path,
    threshold: f64,
) -> Result<usize, Box<dyn std::error::Error>> {
    let reader = BufReader::new(File::open(path)?);
    let mut lines: Vec<String> = Vec::new();
    let mut entries: Vec<Option<TaskEntry>> = Vec::new();

    for line in reader.lines().flatten() {
        let entry = serde_json::from_str::<TaskEntry>(&line).ok();
        entries.push(entry);
        lines.push(line);
    }

    // pass 1: exact duplicates
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

    // pass 2: semantic duplicates via word-trigram jaccard
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
            if jaccard_similarity(trig_a, trig_b) >= threshold {
                keep[kept_indices[j]] = false;
                semantic_dupes += 1;
                break;
            }
        }
    }

    let total_removed = exact_dupes + semantic_dupes;
    if total_removed > 0 {
        let mut f = File::create(path)?;
        for (i, line) in lines.iter().enumerate() {
            if keep[i] {
                f.write_all(line.as_bytes())?;
                f.write_all(b"\n")?;
            }
        }
    }

    Ok(total_removed)
}
