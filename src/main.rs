use crate::analysis::{Data, Dataset, Failure, Provenance, Run};
use crate::config::Config;
use crate::loader::{load_m2, Base};
use crate::openai::Api;
use clap::Parser;
use eyre::Result;
use futures::StreamExt;
use std::path::Path;
use std::time::Duration;

mod analysis;
mod config;
mod loader;
mod openai;
mod scorer;

/// Retries per sentence before it is recorded as a failure.
const RETRIES: usize = 3;

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
    let answers: Vec<Result<Data, Failure>> = futures::stream::iter(challenges)
        .map(|challenge| ask(&api, challenge))
        // Ordered, so that two runs of the same corpus produce comparable files.
        .buffered(config.concurrency.max(1))
        .collect()
        .await;

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

/// One sentence, retried a few times; a sentence that keeps failing is recorded, not fatal.
async fn ask(api: &Api, challenge: Base) -> Result<Data, Failure> {
    let original = challenge.original.clone();
    let failed = |error: eyre::Report| Failure {
        original: original.clone(),
        error: format!("{error:#}"),
    };

    let mut attempt = 0;
    let response = loop {
        match api.send(challenge.original.clone()).await {
            Ok(response) => break response,
            Err(_) if attempt < RETRIES => {
                attempt += 1;
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
            Err(error) => return Err(failed(error)),
        }
    };
    Data::new(challenge, response).map_err(failed)
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
