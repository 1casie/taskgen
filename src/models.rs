use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEntry {
    pub prompt: String,
    pub domain: String,
    pub subdomain: String,
    pub difficulty: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RunStats {
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_tasks: usize,
    pub errors: usize,
}
