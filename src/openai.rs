use reqwest::Client;
use serdev::{Deserialize, Serialize};
use url::Url;

#[derive(Debug)]
pub struct Api {
    api_key: String,
    base: Url,
    model: String,
    client: Client,
}

impl Api {
    pub fn new(api_key: String, base: Url, model: String) -> Api {
        Api {
            api_key,
            base,
            model,
            client: Client::new(),
        }
    }

    pub async fn send(&self, message: String) -> Result<Response, eyre::Report> {
        let request = Request {
            model: self.model.clone(),
            messages: vec![Message {
                content: Some(message),
                role: "user".into(),
            }],
        };
        let response = self
            .client
            .post(self.url())
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await?
            .error_for_status()?;
        Ok(response.json().await?)
    }

    fn url(&self) -> Url {
        let mut url = self.base.clone();
        url.path_segments_mut()
            .unwrap()
            .pop_if_empty()
            .push("chat")
            .push("completions");
        url
    }
}

#[derive(Debug, Serialize)]
struct Request {
    pub model: String,
    pub messages: Vec<Message>,
}

#[derive(Debug, Deserialize)]
pub struct Response {
    pub id: String,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: Message,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: Option<String>,
}
