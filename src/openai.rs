use reqwest::header::{HeaderValue, RETRY_AFTER};
use reqwest::{Client, StatusCode};
use serdev::{Deserialize, Serialize};
use std::fmt;
use std::time::{Duration, SystemTime};
use url::Url;

/// Placeholder in the prompt that is replaced by the challenge sentence.
pub const SENTENCE_PLACEHOLDER: &str = "{sentence}";
pub const DEFAULT_PROMPT: &str = "아래 한국어 문장에서 틀린 부분만 수정하고, 교정 문장 한 줄만 출력하라.\n\
단, 문장이 이미 문법적으로 올바르면 원문을 그대로 출력하라.\n\
문장: {sentence}\n\
교정문:";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// An API failure with enough metadata to decide whether retrying is safe.
#[derive(Debug)]
pub struct ApiError {
    source: reqwest::Error,
    status: Option<StatusCode>,
    retry_after: Option<Duration>,
}

impl ApiError {
    fn request(source: reqwest::Error) -> Self {
        Self {
            source,
            status: None,
            retry_after: None,
        }
    }

    fn response(source: reqwest::Error, status: StatusCode, retry_after: Option<Duration>) -> Self {
        Self {
            source,
            status: Some(status),
            retry_after,
        }
    }

    fn decode(source: reqwest::Error) -> Self {
        Self {
            source,
            status: None,
            retry_after: None,
        }
    }

    pub fn is_retryable(&self) -> bool {
        match self.status {
            Some(status) => is_retryable_status(status),
            None => {
                !self.source.is_builder()
                    && (self.source.is_connect()
                        || self.source.is_timeout()
                        || self.source.is_request())
            }
        }
    }

    pub fn retry_after(&self) -> Option<Duration> {
        self.retry_after
    }
}

impl fmt::Display for ApiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.status {
            Some(status) => write!(formatter, "{status}: {}", self.source),
            None => self.source.fmt(formatter),
        }
    }
}

impl std::error::Error for ApiError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

#[derive(Debug)]
pub struct Api {
    api_key: String,
    base: Url,
    model: String,
    system: Option<String>,
    prompt: String,
    client: Client,
}

impl Api {
    pub fn new(
        api_key: String,
        base: Url,
        model: String,
        system: Option<String>,
        prompt: String,
    ) -> Api {
        Self::with_timeout(api_key, base, model, system, prompt, REQUEST_TIMEOUT)
    }

    fn with_timeout(
        api_key: String,
        base: Url,
        model: String,
        system: Option<String>,
        prompt: String,
        timeout: Duration,
    ) -> Api {
        Api {
            api_key,
            base,
            model,
            system,
            prompt,
            client: Client::builder()
                .timeout(timeout)
                .build()
                .expect("building HTTP client"),
        }
    }

    /// Send one challenge sentence, substituted into the run's prompt template.
    pub async fn send(&self, challenge: String) -> Result<Response, ApiError> {
        let request = self.request(&challenge);
        let response = self
            .client
            .post(self.url())
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await
            .map_err(ApiError::request)?;
        let status = response.status();
        let retry_after = response
            .headers()
            .get(RETRY_AFTER)
            .and_then(parse_retry_after);
        let response = response
            .error_for_status()
            .map_err(|error| ApiError::response(error, status, retry_after))?;
        response.json().await.map_err(ApiError::decode)
    }

    /// The optional system prompt goes first, verbatim; the user message carries the sentence.
    /// Without a system prompt no system message is sent at all, not even an empty one.
    fn request(&self, challenge: &str) -> Request {
        let system = self.system.iter().map(|system| Message {
            role: "system".into(),
            content: Some(system.clone()),
        });
        let user = Message {
            role: "user".into(),
            content: Some(self.prompt.replace(SENTENCE_PLACEHOLDER, challenge)),
        };
        Request {
            model: self.model.clone(),
            messages: system.chain(std::iter::once(user)).collect(),
        }
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
    // pub id: String,
    // pub model: String,
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
    // pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: Option<String>,
}

fn is_retryable_status(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn parse_retry_after(value: &HeaderValue) -> Option<Duration> {
    let value = value.to_str().ok()?.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }

    let retry_at = httpdate::parse_http_date(value).ok()?;
    retry_at.duration_since(SystemTime::now()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::HeaderValue;
    use reqwest::StatusCode;
    use std::time::{Duration, SystemTime};

    #[test]
    fn parses_retry_after_seconds() {
        let value = HeaderValue::from_static("2");

        assert_eq!(parse_retry_after(&value), Some(Duration::from_secs(2)));
    }

    #[test]
    fn parses_retry_after_http_date() {
        let value = HeaderValue::from_str(&httpdate::fmt_http_date(
            SystemTime::now() + Duration::from_secs(5),
        ))
        .unwrap();

        assert!(parse_retry_after(&value).is_some_and(|delay| delay <= Duration::from_secs(5)));
    }

    #[test]
    fn ignores_invalid_retry_after() {
        let value = HeaderValue::from_static("not-a-delay");

        assert_eq!(parse_retry_after(&value), None);
    }

    #[test]
    fn retries_only_throttling_and_server_statuses() {
        assert!(is_retryable_status(StatusCode::TOO_MANY_REQUESTS));
        assert!(is_retryable_status(StatusCode::INTERNAL_SERVER_ERROR));
        assert!(is_retryable_status(StatusCode::BAD_GATEWAY));
        assert!(!is_retryable_status(StatusCode::BAD_REQUEST));
        assert!(!is_retryable_status(StatusCode::UNAUTHORIZED));
    }

    #[test]
    fn substitutes_the_sentence_into_a_single_user_message() {
        let api = Api::new(
            "test-key".into(),
            Url::parse("http://localhost").unwrap(),
            "test-model".into(),
            None,
            DEFAULT_PROMPT.into(),
        );

        let request = api.request("저는 학교에 갔어요.");

        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.messages[0].role, "user");
        assert_eq!(
            request.messages[0].content.as_deref(),
            Some(
                "아래 한국어 문장에서 틀린 부분만 수정하고, 교정 문장 한 줄만 출력하라.\n\
                 단, 문장이 이미 문법적으로 올바르면 원문을 그대로 출력하라.\n\
                 문장: 저는 학교에 갔어요.\n\
                 교정문:"
            )
        );
    }

    #[test]
    fn sends_the_system_prompt_first_and_the_sentence_as_user_message() {
        let api = Api::new(
            "test-key".into(),
            Url::parse("http://localhost").unwrap(),
            "test-model".into(),
            Some("Correct the Korean sentence.".into()),
            "{sentence}".into(),
        );

        let request = api.request("저는 학교에 갔어요.");

        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0].role, "system");
        assert_eq!(
            request.messages[0].content.as_deref(),
            Some("Correct the Korean sentence.")
        );
        assert_eq!(request.messages[1].role, "user");
        assert_eq!(
            request.messages[1].content.as_deref(),
            Some("저는 학교에 갔어요.")
        );
    }

    #[tokio::test]
    async fn send_times_out_when_the_endpoint_never_responds() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (_stream, _) = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_secs(1)).await;
        });
        let api = Api::with_timeout(
            "test-key".into(),
            Url::parse(&format!("http://{address}")).unwrap(),
            "test-model".into(),
            None,
            DEFAULT_PROMPT.into(),
            Duration::from_millis(50),
        );

        let result = tokio::time::timeout(Duration::from_secs(1), api.send("test".into()))
            .await
            .expect("request should time out before the test timeout");

        assert!(result.unwrap_err().is_retryable());
        server.abort();
    }
}
