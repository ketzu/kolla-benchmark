use serdev::Deserialize;
use std::str::FromStr;

#[derive(Debug, Deserialize)]
pub struct Base {
    pub original: String,
    pub corrections: Vec<String>,
}

impl Base {
    pub fn new(original: String, corrections: Vec<String>) -> Self {
        Self {
            original,
            corrections,
        }
    }
}
