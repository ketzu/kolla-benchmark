use crate::analysis::{Data, Dataset, Failure, Provenance, Round, Run};
use crate::config::Config;
use crate::loader::{Base, load_m2};
use crate::openai::{Api, Response};
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
    if let Some(max_iterations) = config.iterations() {
        println!("Sending every answer back, up to {max_iterations} requests per sentence");
    }

    let prompt = config.prompt();
    let api = Api::new(
        key,
        config.url.clone(),
        model.clone(),
        config.system.clone(),
        prompt.clone(),
    );
    let total = challenges.len();
    let breaker = CircuitBreaker::default();
    let progress = Progress::new(total, &model);
    let (api, breaker, progress) = (&api, &breaker, &progress);
    let originals = challenges
        .iter()
        .map(|challenge| challenge.original.as_str());
    let answers = match config.iterations() {
        None => {
            saturate(
                originals,
                config.concurrency,
                |index, original| async move {
                    let _worker = progress.start();
                    ask(api, original.to_owned(), index, total, breaker, progress)
                        .await
                        .map(|result| result.map(Answer::Once))
                },
            )
            .await
        }
        Some(max_iterations) => {
            saturate(
                originals,
                config.concurrency,
                |index, original| async move {
                    let _worker = progress.start();
                    iterate(original, max_iterations, |sentence| {
                        ask(api, sentence, index, total, breaker, progress)
                    })
                    .await
                },
            )
            .await
        }
    };
    progress.finish(breaker.is_open());

    // Scored only once every request is done, so that scoring never holds up the network.
    let mut results = Vec::with_capacity(total);
    let mut failures = Vec::new();
    for (index, (challenge, answer)) in challenges.into_iter().zip(answers).enumerate() {
        match answer {
            // Never answered: the run stopped first.
            None => {}
            Some(Err(failure)) => failures.push(failure),
            Some(Ok(answer)) => {
                let original = challenge.original.clone();
                let data = match answer {
                    Answer::Once(response) => Data::new(challenge, response),
                    Answer::Iterated { rounds, converged } => {
                        Ok(Data::iterated(challenge, rounds, converged))
                    }
                };
                match data {
                    Ok(data) => results.push(data),
                    Err(error) => {
                        eprintln!(
                            "[{}/{}] response could not be scored and will not be retried: {}",
                            index + 1,
                            total,
                            error
                        );
                        failures.push(failure(&original, error));
                    }
                }
            }
        }
    }

    let provenance = Provenance::new(
        model,
        config.url.to_string(),
        config.system.clone(),
        prompt,
        config.iterations(),
        dataset,
    );
    Ok(Run::new(provenance, results, failures))
}

/// What came back for one sentence, before it is scored.
#[derive(Debug)]
enum Answer {
    Once(Response),
    Iterated { rounds: Vec<Round>, converged: bool },
}

/// Send the sentence, then every answer back in its place, until an answer tokenizes the same as
/// the text it was sent or `max_iterations` requests were made. A request that fails fails the
/// whole sentence, which keeps the rounds that came back before it.
async fn iterate<Fut>(
    original: &str,
    max_iterations: usize,
    mut ask: impl FnMut(String) -> Fut,
) -> Outcome<Result<Answer, Failure>>
where
    Fut: Future<Output = Outcome<Result<Response, Failure>>>,
{
    let mut rounds = Vec::new();
    let mut sent = original.to_owned();
    loop {
        let Outcome { result, stop } = ask(sent.clone()).await;
        let round = result
            .map_err(|failure| failure.error)
            .and_then(|response| Round::of(response).map_err(|error| error.to_string()));
        let round = match round {
            Ok(round) => round,
            Err(error) => {
                return Outcome {
                    result: Err(Failure {
                        original: original.to_owned(),
                        error,
                        rounds,
                    }),
                    stop,
                };
            }
        };

        let converged = scorer::tokenize(&round.answer) == scorer::tokenize(&sent);
        sent = round.answer.clone();
        rounds.push(round);
        if converged || rounds.len() >= max_iterations {
            return Outcome {
                result: Ok(Answer::Iterated { rounds, converged }),
                stop,
            };
        }
    }
}

/// Run `work` on every item with `concurrency` items in flight for as long as there are items
/// left: the moment any one finishes, the next one starts. Ordered buffering would not do that —
/// a finished item waiting for a slower one before it keeps holding its slot.
///
/// Answers come back in item order. The run stops after the first outcome that asks for it,
/// dropping the work still in flight; items that never finished answer `None`.
async fn saturate<T, O, Fut>(
    items: impl ExactSizeIterator<Item = T>,
    concurrency: usize,
    mut work: impl FnMut(usize, T) -> Fut,
) -> Vec<Option<O>>
where
    Fut: Future<Output = Outcome<O>>,
{
    let mut answers: Vec<Option<O>> = (0..items.len()).map(|_| None).collect();
    let mut outcomes = futures::stream::iter(items.enumerate())
        .map(|(index, item)| {
            let outcome = work(index, item);
            async move { (index, outcome.await) }
        })
        .buffer_unordered(concurrency.max(1));
    while let Some((index, outcome)) = outcomes.next().await {
        answers[index] = Some(outcome.result);
        if outcome.stop {
            break;
        }
    }
    answers
}

/// One request for a sentence, retried a few times; repeated retry exhaustion stops the run.
async fn ask(
    api: &Api,
    original: String,
    index: usize,
    total: usize,
    breaker: &CircuitBreaker,
    progress: &Progress,
) -> Outcome<Result<Response, Failure>> {
    let mut retries = 0;
    loop {
        if breaker.is_open() {
            progress.log(format!(
                "[{}/{}] not retried because the benchmark circuit is open",
                index + 1,
                total
            ));
            return Outcome {
                result: Err(failure(
                    &original,
                    "benchmark stopped after repeated retry-exhausted failures",
                )),
                stop: true,
            };
        }

        let attempt = retries + 1;
        match api.send(original.to_owned()).await {
            Ok(response) => {
                breaker.record_success();
                return Outcome {
                    result: Ok(response),
                    stop: false,
                };
            }
            Err(error) if !error.is_retryable() => {
                progress.log(format!(
                    "[{}/{}] request failed on attempt {} and will not be retried: {}",
                    index + 1,
                    total,
                    attempt,
                    error
                ));
                return Outcome {
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
                return Outcome {
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
                // The sentence keeps its slot while it waits: an endpoint that asks us to back
                // off should see fewer requests, not the same number from other sentences.
                let _waiting = progress.back_off();
                tokio::time::sleep(delay).await;
            }
        }
    }
}

fn failure(original: &str, error: impl std::fmt::Display) -> Failure {
    Failure {
        original: original.to_owned(),
        error: error.to_string(),
        rounds: Vec::new(),
    }
}

struct Outcome<T> {
    result: T,
    stop: bool,
}

impl<T> Outcome<T> {
    fn map<U>(self, map: impl FnOnce(T) -> U) -> Outcome<U> {
        Outcome {
            result: map(self.result),
            stop: self.stop,
        }
    }
}

#[derive(Clone)]
struct Progress {
    bar: ProgressBar,
    /// Sentences started and not yet finished, including those waiting out a backoff.
    active: Arc<AtomicUsize>,
    backing_off: Arc<AtomicUsize>,
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
            active: Arc::new(AtomicUsize::new(0)),
            backing_off: Arc::new(AtomicUsize::new(0)),
            total,
        }
    }

    fn start(&self) -> ProgressTask {
        self.active.fetch_add(1, Ordering::SeqCst);
        self.update_message();
        ProgressTask {
            progress: self.clone(),
        }
    }

    fn complete(&self) {
        self.active.fetch_sub(1, Ordering::SeqCst);
        self.bar.inc(1);
        self.update_message();
    }

    fn back_off(&self) -> BackingOff {
        self.backing_off.fetch_add(1, Ordering::SeqCst);
        self.update_message();
        BackingOff {
            progress: self.clone(),
        }
    }

    fn update_message(&self) {
        let backing_off = self.backing_off.load(Ordering::SeqCst);
        let in_flight = self
            .active
            .load(Ordering::SeqCst)
            .saturating_sub(backing_off);
        let message = match backing_off {
            0 => format!("{in_flight} in flight"),
            _ => format!("{in_flight} in flight, {backing_off} backing off"),
        };
        self.bar.set_message(message);
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
        self.active.load(Ordering::SeqCst) - self.backing_off.load(Ordering::SeqCst)
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

struct BackingOff {
    progress: Progress,
}

impl Drop for BackingOff {
    fn drop(&mut self) {
        self.progress.backing_off.fetch_sub(1, Ordering::SeqCst);
        self.progress.update_message();
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
            {
                let _waiting = progress.back_off();
                assert_eq!(progress.in_flight(), 0);
            }
            assert_eq!(progress.in_flight(), 1);
        }

        assert_eq!(progress.in_flight(), 0);
        assert_eq!(progress.position(), 1);
    }

    fn done<T>(result: T) -> Outcome<T> {
        Outcome {
            result,
            stop: false,
        }
    }

    #[tokio::test]
    async fn saturate_refills_slots_while_an_earlier_item_is_still_running() {
        let started = AtomicUsize::new(0);
        let started_when_first_finished = AtomicUsize::new(0);

        let answers = saturate(0..12, 3, |index, item| {
            let (started, first) = (&started, &started_when_first_finished);
            async move {
                started.fetch_add(1, Ordering::SeqCst);
                let millis = if index == 0 { 400 } else { 10 };
                tokio::time::sleep(Duration::from_millis(millis)).await;
                if index == 0 {
                    first.store(started.load(Ordering::SeqCst), Ordering::SeqCst);
                }
                done(item * 10)
            }
        })
        .await;

        // Ordered buffering would have started only the first three before the first finished.
        assert_eq!(started_when_first_finished.load(Ordering::SeqCst), 12);
        assert_eq!(
            answers,
            (0..12).map(|item| Some(item * 10)).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn saturate_never_exceeds_the_concurrency() {
        let running = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);

        saturate(0..40, 4, |index, _| {
            let (running, peak) = (&running, &peak);
            async move {
                let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(1 + (index as u64 * 7) % 13)).await;
                running.fetch_sub(1, Ordering::SeqCst);
                done(())
            }
        })
        .await;

        assert_eq!(peak.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn saturate_stops_after_the_outcome_that_asks_for_it() {
        let answers = saturate(0..5, 1, |index, item| async move {
            Outcome {
                result: item,
                stop: index == 2,
            }
        })
        .await;

        assert_eq!(answers, vec![Some(0), Some(1), Some(2), None, None]);
    }

    fn reply(answer: &str) -> Outcome<Result<Response, Failure>> {
        done(Ok(Response {
            choices: vec![crate::openai::Choice {
                message: crate::openai::Message {
                    role: "assistant".into(),
                    content: Some(answer.into()),
                },
            }],
            usage: None,
        }))
    }

    /// Iterate `original` against answers given in order, returning what was sent too.
    async fn iterate_over(
        original: &str,
        max_iterations: usize,
        replies: Vec<Outcome<Result<Response, Failure>>>,
    ) -> (Outcome<Result<Answer, Failure>>, Vec<String>) {
        let mut sent = Vec::new();
        let mut replies = replies.into_iter();
        let outcome = iterate(original, max_iterations, |sentence| {
            sent.push(sentence);
            let reply = replies.next().expect("asked more often than expected");
            async move { reply }
        })
        .await;
        (outcome, sent)
    }

    fn answers(rounds: &[Round]) -> Vec<&str> {
        rounds.iter().map(|round| round.answer.as_str()).collect()
    }

    #[tokio::test]
    async fn iterate_stops_when_the_first_answer_is_the_sentence_unchanged() {
        // Only the space before the full stop differs, which tokenizes away.
        let (outcome, sent) = iterate_over(
            "우리는 배가 고펐습니다 .",
            10,
            vec![reply("우리는 배가 고펐습니다.")],
        )
        .await;

        let Ok(Answer::Iterated { rounds, converged }) = outcome.result else {
            panic!("expected an iterated answer");
        };
        assert!(converged);
        assert_eq!(answers(&rounds), ["우리는 배가 고펐습니다."]);
        assert_eq!(sent, ["우리는 배가 고펐습니다 ."]);
    }

    #[tokio::test]
    async fn iterate_sends_every_answer_back_until_one_repeats() {
        let (outcome, sent) = iterate_over("a", 10, vec![reply("b"), reply("c"), reply("c")]).await;

        let Ok(Answer::Iterated { rounds, converged }) = outcome.result else {
            panic!("expected an iterated answer");
        };
        assert!(converged);
        assert_eq!(answers(&rounds), ["b", "c", "c"]);
        assert_eq!(sent, ["a", "b", "c"]);
    }

    #[tokio::test]
    async fn iterate_gives_up_after_max_iterations() {
        let (outcome, sent) = iterate_over("a", 2, vec![reply("b"), reply("a")]).await;

        let Ok(Answer::Iterated { rounds, converged }) = outcome.result else {
            panic!("expected an iterated answer");
        };
        assert!(!converged);
        assert_eq!(answers(&rounds), ["b", "a"]);
        assert_eq!(sent.len(), 2);
    }

    #[tokio::test]
    async fn a_failed_round_fails_the_sentence_and_keeps_the_rounds_before_it() {
        let failed = Outcome {
            result: Err(failure("b", "boom")),
            stop: true,
        };
        let (outcome, _) = iterate_over("a", 10, vec![reply("b"), failed]).await;

        assert!(outcome.stop);
        let Err(failure) = outcome.result else {
            panic!("expected a failure");
        };
        assert_eq!(failure.original, "a");
        assert_eq!(failure.error, "boom");
        assert_eq!(answers(&failure.rounds), ["b"]);
    }
}
