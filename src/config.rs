use clap::Parser;
use url::Url;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Config {
    /// API Key to send along with requests
    #[arg(long, env)]
    pub api_key: String,
    /// Model to evaluate
    #[arg(short, long)]
    pub model: String,
    /// Base URL for OpenAI compatible request
    #[arg(short, long, default_value = "https://openrouter.ai/api/v1")]
    pub url: Url,
}
