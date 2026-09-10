use crate::analysis::Data;
use crate::config::Config;
use crate::loader::load_m2;
use crate::openai::Api;
use clap::Parser;
use futures;
use futures::{StreamExt, TryStreamExt};
use std::time::Duration;
use tokio::main;

mod analysis;
mod config;
mod loader;
mod openai;

#[main]
async fn main() {
    let config = Config::parse();

    println!(
        "Starting evaluation on {} against {}",
        config.model, config.url
    );

    let openai = Api::new(config.api_key, config.url, config.model);

    let kolla = load_m2("data/KoLLA_multi-refs.m2").expect("Failed to load Kolla data.");

    let responses = futures::stream::iter(kolla)
        .map(async |request| {
            let mut attempt = 0;
            let response = loop {
                match openai.send(request.original.clone()).await {
                    Ok(response) => break response,
                    Err(e) if attempt < 3 => {
                        attempt += 1;
                        tokio::time::sleep(Duration::from_millis(250)).await;
                    }
                    Err(e) => return Err(e),
                }
            };
            Data::new(request, response)
        })
        .buffer_unordered(10)
        .try_collect::<Vec<_>>()
        .await;
}
