use crate::loader::Base;
use crate::openai::Response;
use eyre::ContextCompat;
use serdev::Serialize;

#[derive(Debug, Serialize)]
pub struct Data {
    request: String,
    answer: String,
    gold: Vec<String>,
    prompt: u64,
    result: u64,
    total: u64,
}

impl Data {
    pub fn new(base: Base, response: Response) -> Result<Data, eyre::Report> {
        let answer = response
            .choices
            .into_iter()
            .next()
            .context("Missing Choice")?
            .message
            .content
            .context("Missing answer")?;
        let usage = response.usage.context("Missing Usage")?;

        Ok(Data {
            request: base.original,
            gold: base.corrections,
            answer,
            prompt: usage.prompt_tokens,
            result: usage.completion_tokens,
            total: usage.total_tokens,
        })
    }
}
