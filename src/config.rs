use crate::openai::DEFAULT_PROMPT;
use clap::Parser;
use std::path::PathBuf;
use url::Url;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Config {
    /// API Key to send along with requests
    #[arg(long, env, required_unless_present_any = ["rescore", "baseline"])]
    pub api_key: Option<String>,
    /// Model to evaluate
    #[arg(short, long, required_unless_present_any = ["rescore", "baseline"])]
    pub model: Option<String>,
    /// Base URL for OpenAI compatible request
    #[arg(short, long, default_value = "https://openrouter.ai/api/v1")]
    pub url: Url,
    /// Instruction the challenge sentence is wrapped in
    #[arg(long, default_value = DEFAULT_PROMPT)]
    pub prompt: String,
    /// KoLLA M2 annotations to evaluate against
    #[arg(short, long, default_value = "data/KoLLA_multi-refs.m2")]
    pub data: PathBuf,
    /// Sentences to evaluate; 0 runs the whole corpus (that costs real money)
    #[arg(short, long, default_value_t = 25)]
    pub limit: usize,
    /// Requests in flight at the same time
    #[arg(short, long, default_value_t = 10)]
    pub concurrency: usize,
    /// Where to write the run; defaults to results/<model>-<timestamp>.json
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Score a previously written run again instead of calling the API
    #[arg(long, conflicts_with_all = ["api_key", "model"])]
    pub rescore: Option<PathBuf>,
    /// Score the corpus against itself instead of calling the API
    #[arg(long, conflicts_with_all = ["api_key", "model", "rescore"])]
    pub baseline: bool,
}
