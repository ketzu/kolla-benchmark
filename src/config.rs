use crate::openai::{DEFAULT_PROMPT, EXTENDED_PROMPT, SENTENCE_PLACEHOLDER};
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
    /// System prompt sent before the user message; no system message is sent without it
    #[arg(long)]
    pub system: Option<String>,
    /// User prompt template; {sentence} is replaced by the challenge sentence [default: the
    /// Korean prompt, or the extended prompt with --iterate]
    #[arg(long, value_parser = prompt_template)]
    prompt: Option<String>,
    /// KoLLA M2 annotations to evaluate against
    #[arg(short, long, default_value = "data/KoLLA_multi-refs.m2")]
    pub data: PathBuf,
    /// Sentences to evaluate; 0 runs the whole corpus (that costs real money)
    #[arg(short, long, default_value_t = 25)]
    pub limit: usize,
    /// Requests in flight at the same time
    #[arg(short, long, default_value_t = 10)]
    pub concurrency: usize,
    /// Send every answer back as the sentence to correct until the model returns it unchanged
    #[arg(long, conflicts_with_all = ["rescore", "baseline"])]
    pub iterate: bool,
    /// Requests per sentence before an iterated sentence stops without settling
    #[arg(long, default_value_t = 10, requires = "iterate", value_parser = at_least_one)]
    pub max_iterations: usize,
    /// Where to write the run; defaults to results/<model>-<timestamp>.json, or to
    /// iterate-results/ with --iterate
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Score a previously written run again instead of calling the API
    #[arg(long, conflicts_with_all = ["api_key", "model"])]
    pub rescore: Option<PathBuf>,
    /// Score the corpus against itself instead of calling the API
    #[arg(long, conflicts_with_all = ["api_key", "model", "rescore"])]
    pub baseline: bool,
}

impl Config {
    /// The prompt template to send: the one given, else the default of the experiment.
    pub fn prompt(&self) -> String {
        match (&self.prompt, self.iterate) {
            (Some(prompt), _) => prompt.clone(),
            (None, false) => DEFAULT_PROMPT.to_owned(),
            (None, true) => EXTENDED_PROMPT.to_owned(),
        }
    }

    /// The most requests an iterated sentence may take; none when every sentence is asked once.
    pub fn iterations(&self) -> Option<usize> {
        self.iterate.then_some(self.max_iterations)
    }
}

/// A prompt without the placeholder would never show the model the sentence.
fn prompt_template(prompt: &str) -> Result<String, String> {
    if prompt.contains(SENTENCE_PLACEHOLDER) {
        Ok(prompt.to_owned())
    } else {
        Err(format!("the prompt must contain {SENTENCE_PLACEHOLDER}"))
    }
}

/// An iterated sentence is asked at least once.
fn at_least_one(value: &str) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(0) => Err("at least one request is needed".to_owned()),
        Ok(count) => Ok(count),
        Err(error) => Err(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Config, clap::Error> {
        let base = ["kolla-benchmark", "--api-key", "key", "--model", "model"];
        Config::try_parse_from(base.iter().chain(args))
    }

    #[test]
    fn the_default_prompt_depends_on_the_experiment() {
        let single = parse(&[]).unwrap();
        assert_eq!(single.prompt(), DEFAULT_PROMPT);
        assert_eq!(single.iterations(), None);

        let iterated = parse(&["--iterate"]).unwrap();
        assert_eq!(iterated.prompt(), EXTENDED_PROMPT);
        assert_eq!(iterated.iterations(), Some(10));
    }

    #[test]
    fn a_given_prompt_wins_over_the_default() {
        let config = parse(&["--iterate", "--prompt", "Fix: {sentence}"]).unwrap();
        assert_eq!(config.prompt(), "Fix: {sentence}");
    }

    #[test]
    fn max_iterations_needs_iterate_and_at_least_one_request() {
        assert!(parse(&["--max-iterations", "3"]).is_err());
        assert!(parse(&["--iterate", "--max-iterations", "0"]).is_err());
        assert_eq!(
            parse(&["--iterate", "--max-iterations", "3"])
                .unwrap()
                .iterations(),
            Some(3)
        );
    }
}
