use crate::analysis::{Data, Dataset, Failure, Provenance, Run};
use crate::config::Config;
use crate::loader::{Base, load_m2};
use crate::openai::Api;
use clap::Parser;
use eyre::Result;
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use rand::RngExt;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

mod analysis;
mod config;
mod loader;
mod openai;
pub mod python_scorer;
#[allow(dead_code)]
mod scorer;

/// Retries per sentence before it is recorded as a failure.
const RETRIES: usize = 3;
/// Stop the benchmark when the endpoint keeps rejecting requests after retries.
const MAX_CONSECUTIVE_FAILURES: usize = 3;
const INITIAL_BACKOFF: Duration = Duration::from_millis(500);
const MAX_BACKOFF: Duration = Duration::from_secs(30);
const MAX_JITTER: u64 = 5_000;

#[tokio::main]
async fn main() -> Result<()> {
    let config = Config::parse();

    if config.baseline {
        println!("{}", analysis::baselines(&load_m2(&config.data)?));
        return Ok(());
    }

    let run = match &config.rescore {
        Some(path) => rescore(path)?,
        None => benchmark(&config).await?,
    };
    println!("{}", run.report());

    // A re-score only writes when a destination was asked for; a benchmark always writes,
    // it cost money to produce.
    if config.rescore.is_none() || config.output.is_some() {
        let output = config
            .output
            .clone()
            .unwrap_or_else(|| run.default_output());
        run.write(&output)?;
        println!("written to {}", output.display());
    }
    Ok(())
}

/// Ask the model for every sentence, then score what came back.
async fn benchmark(config: &Config) -> Result<Run> {
    let corpus = load_m2(&config.data)?;
    let dataset = Dataset::of(&config.data, corpus.len())?;
    let model = config.model.clone().expect("required unless --rescore");
    let key = config.api_key.clone().expect("required unless --rescore");

    let challenges: Vec<Base> = match config.limit {
        0 => corpus,
        limit => corpus.into_iter().take(limit).collect(),
    };
    println!(
        "Evaluating {model} on {} of {} sentences against {}",
        challenges.len(),
        dataset.sentences,
        config.url
    );

    let api = Api::new(
        key,
        config.url.clone(),
        model.clone(),
        config.prompt.clone(),
    );
    let total = challenges.len();
    let breaker = Arc::new(CircuitBreaker::default());
    let progress = Progress::new(total, &model);
    let api_ref = &api;
    let progress_ref = progress.clone();
    let answers: Vec<Result<Data, Failure>> =
        futures::stream::iter(challenges.into_iter().enumerate())
            .map(|(index, challenge)| {
                let breaker = Arc::clone(&breaker);
                let progress = progress_ref.clone();
                async move { ask(api_ref, challenge, index, total, breaker, progress).await }
            })
            // Ordered, so that two runs of the same corpus produce comparable files.
            .buffered(config.concurrency.max(1))
            // Include the item that opened the circuit, then drop queued and in-flight work.
            .scan(false, |stopped, outcome| {
                if *stopped {
                    return std::future::ready(None);
                }
                if outcome.stop {
                    *stopped = true;
                }
                std::future::ready(Some(outcome.result))
            })
            .collect()
            .await;
    progress.finish(breaker.is_open());

    let (results, failures): (Vec<_>, Vec<_>) = answers.into_iter().partition(Result::is_ok);
    let provenance = Provenance::new(
        model,
        config.url.to_string(),
        config.prompt.clone(),
        dataset,
    );
    Ok(Run::new(
        provenance,
        results.into_iter().map(Result::unwrap).collect(),
        failures.into_iter().map(Result::unwrap_err).collect(),
    ))
}

/// One sentence, retried a few times; repeated retry exhaustion stops the run.
async fn ask(
    api: &Api,
    challenge: Base,
    index: usize,
    total: usize,
    breaker: Arc<CircuitBreaker>,
    progress: Progress,
) -> AskOutcome {
    let _worker = progress.start();
    let original = challenge.original.clone();
    let mut retries = 0;
    let response = loop {
        if breaker.is_open() {
            progress.log(format!(
                "[{}/{}] not retried because the benchmark circuit is open",
                index + 1,
                total
            ));
            return AskOutcome {
                result: Err(Failure {
                    original,
                    error: "benchmark stopped after repeated retry-exhausted failures".into(),
                }),
                stop: true,
            };
        }

        let attempt = retries + 1;
        match api.send(challenge.original.clone()).await {
            Ok(response) => {
                breaker.record_success();
                break response;
            }
            Err(error) if !error.is_retryable() => {
                progress.log(format!(
                    "[{}/{}] request failed on attempt {} and will not be retried: {}",
                    index + 1,
                    total,
                    attempt,
                    error
                ));
                return AskOutcome {
                    result: Err(failure(&original, error)),
                    stop: false,
                };
            }
            Err(error) if retries >= RETRIES => {
                let opened_now = breaker.record_retry_exhaustion();
                progress.log(format!(
                    "[{}/{}] request failed after {} attempts: {}",
                    index + 1,
                    total,
                    attempt,
                    error
                ));
                if opened_now {
                    progress.log(format!(
                        "stopping benchmark after {} consecutive retry-exhausted failures",
                        MAX_CONSECUTIVE_FAILURES
                    ));
                }
                return AskOutcome {
                    result: Err(failure(&original, error)),
                    stop: opened_now,
                };
            }
            Err(error) => {
                retries += 1;
                let delay = next_retry_delay(retries, error.retry_after());
                progress.log(format!(
                    "[{}/{}] request failed on attempt {}: {}; retry {}/{} in {:?}",
                    index + 1,
                    total,
                    attempt,
                    error,
                    retries,
                    RETRIES,
                    delay
                ));
                tokio::time::sleep(delay).await;
            }
        }
    };

    match Data::new(challenge, response) {
        Ok(data) => AskOutcome {
            result: Ok(data),
            stop: false,
        },
        Err(error) => {
            progress.log(format!(
                "[{}/{}] response could not be scored and will not be retried: {}",
                index + 1,
                total,
                error
            ));
            AskOutcome {
                result: Err(failure(&original, error)),
                stop: false,
            }
        }
    }
}

fn failure(original: &str, error: impl std::fmt::Display) -> Failure {
    Failure {
        original: original.to_owned(),
        error: error.to_string(),
    }
}

struct AskOutcome {
    result: Result<Data, Failure>,
    stop: bool,
}

#[derive(Clone)]
struct Progress {
    bar: ProgressBar,
    in_flight: Arc<AtomicUsize>,
    total: u64,
}

impl Progress {
    fn new(total: usize, model: &str) -> Self {
        let total = total as u64;
        let bar = ProgressBar::new(total);
        bar.set_style(
            ProgressStyle::with_template(
                "{prefix:.bold} {bar:40.cyan/blue} {pos}/{len} {elapsed_precise} {msg}",
            )
            .expect("valid progress bar template"),
        );
        bar.set_prefix(model.to_owned());
        bar.enable_steady_tick(Duration::from_secs(1));
        Self {
            bar,
            in_flight: Arc::new(AtomicUsize::new(0)),
            total,
        }
    }

    fn start(&self) -> ProgressTask {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        self.update_message();
        ProgressTask {
            progress: self.clone(),
        }
    }

    fn complete(&self) {
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        self.bar.inc(1);
        self.update_message();
    }

    fn update_message(&self) {
        self.bar.set_message(format!(
            "{} in flight",
            self.in_flight.load(Ordering::SeqCst)
        ));
    }

    fn log(&self, message: String) {
        self.bar.suspend(|| eprintln!("{message}"));
    }

    fn finish(&self, stopped: bool) {
        let message = if stopped {
            format!("stopped at {}/{}", self.bar.position(), self.total)
        } else {
            "complete".to_owned()
        };
        self.bar.finish_with_message(message);
    }

    #[cfg(test)]
    fn position(&self) -> u64 {
        self.bar.position()
    }

    #[cfg(test)]
    fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::SeqCst)
    }
}

struct ProgressTask {
    progress: Progress,
}

impl Drop for ProgressTask {
    fn drop(&mut self) {
        self.progress.complete();
    }
}

#[derive(Debug, Default)]
struct CircuitBreaker {
    consecutive_failures: AtomicUsize,
    open: AtomicBool,
}

impl CircuitBreaker {
    fn is_open(&self) -> bool {
        self.open.load(Ordering::SeqCst)
    }

    fn record_success(&self) {
        if !self.is_open() {
            self.consecutive_failures.store(0, Ordering::SeqCst);
        }
    }

    fn record_retry_exhaustion(&self) -> bool {
        let failures = self.consecutive_failures.fetch_add(1, Ordering::SeqCst) + 1;
        if failures < MAX_CONSECUTIVE_FAILURES {
            return false;
        }
        !self.open.swap(true, Ordering::SeqCst)
    }
}

fn fallback_backoff(retry_number: usize) -> Duration {
    let exponent = retry_number.saturating_sub(1).min(6);
    let multiplier = 1u64 << exponent;
    let millis = INITIAL_BACKOFF
        .as_millis()
        .saturating_mul(u128::from(multiplier))
        .min(MAX_BACKOFF.as_millis());
    Duration::from_millis(millis as u64)
}

fn retry_delay(retry_number: usize, retry_after: Option<Duration>, jitter_ms: u64) -> Duration {
    let base = fallback_backoff(retry_number).max(retry_after.unwrap_or_default());
    base.saturating_add(Duration::from_millis(jitter_ms))
}

fn next_retry_delay(retry_number: usize, retry_after: Option<Duration>) -> Duration {
    let base = fallback_backoff(retry_number).max(retry_after.unwrap_or_default());
    let jitter_limit = (base.as_millis() as u64 / 4).min(MAX_JITTER);
    let jitter_ms = rand::rng().random_range(0..=jitter_limit);
    retry_delay(retry_number, retry_after, jitter_ms)
}

/// Score the answers of an earlier run again, without paying for them twice.
fn rescore(path: &Path) -> Result<Run> {
    let mut run = Run::read(path)?;
    let mut results = std::mem::take(&mut run.results);
    results.iter_mut().for_each(Data::rescore);

    let mut provenance = run.provenance.clone();
    provenance.rescored = Some(analysis::utc(analysis::unix_now()));
    Ok(Run::new(provenance, results, run.failures))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circuit_breaker_opens_after_three_retry_exhaustions() {
        let breaker = CircuitBreaker::default();

        assert!(!breaker.record_retry_exhaustion());
        assert!(!breaker.record_retry_exhaustion());
        assert!(breaker.record_retry_exhaustion());
        assert!(breaker.is_open());
    }

    #[test]
    fn successful_request_resets_the_retry_exhaustion_streak() {
        let breaker = CircuitBreaker::default();

        assert!(!breaker.record_retry_exhaustion());
        breaker.record_success();
        assert!(!breaker.record_retry_exhaustion());
    }

    #[test]
    fn fallback_backoff_grows_exponentially_and_is_capped() {
        assert_eq!(fallback_backoff(1), Duration::from_millis(500));
        assert_eq!(fallback_backoff(2), Duration::from_secs(1));
        assert_eq!(fallback_backoff(3), Duration::from_secs(2));
        assert_eq!(fallback_backoff(20), MAX_BACKOFF);
    }

    #[test]
    fn retry_after_is_used_as_the_minimum_delay() {
        assert_eq!(
            retry_delay(1, Some(Duration::from_secs(5)), 0),
            Duration::from_secs(5)
        );
        assert_eq!(
            retry_delay(2, Some(Duration::from_millis(100)), 0),
            Duration::from_secs(1)
        );
        assert_eq!(retry_delay(1, None, 125), Duration::from_millis(625));
    }

    #[test]
    fn jittered_backoff_stays_within_the_expected_range() {
        for retry_number in 1..=3 {
            let base = fallback_backoff(retry_number);
            let delay = next_retry_delay(retry_number, None);

            assert!(delay >= base);
            assert!(delay <= base + Duration::from_millis(base.as_millis() as u64 / 4));
        }
    }

    #[test]
    fn progress_advances_when_a_worker_finishes() {
        let progress = Progress::new(1, "test");

        {
            let _worker = progress.start();
            assert_eq!(progress.in_flight(), 1);
            assert_eq!(progress.position(), 0);
        }

        assert_eq!(progress.in_flight(), 0);
        assert_eq!(progress.position(), 1);
    }
}
