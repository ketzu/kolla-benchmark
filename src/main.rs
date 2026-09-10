use crate::analysis::Data;
use crate::config::Config;
use crate::loader::Base;
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

    let requests: Vec<Base> = vec![
        Base::new(
            "Respond with 'Hello' and nothing else.".into(),
            vec!["Hello".into(), "Hello".into()],
        ),
        Base::new(
            "Respond with 'world' and nothing else.".into(),
            vec!["World".into(), "World".into()],
        ),
    ];

    let responses = futures::stream::iter(requests)
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
