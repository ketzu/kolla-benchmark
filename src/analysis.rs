//! What a run produced, what it was produced with, and how well it did.

use crate::loader::Base;
use crate::openai::Response;
use crate::python_scorer;
use crate::scorer::{self, Counts, Metrics, SentenceScore};
use eyre::{Context, ContextCompat, Result};
use serdev::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// One sentence: the gold standard item, what the model answered, and how it scored.
/// The item is kept verbatim so that a written run can be scored again without the corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Data {
    pub base: Base,
    pub answer: String,
    pub answer_tokens: Vec<String>,
    pub score: SentenceScore,
    pub usage: Usage,
}

impl Data {
    pub fn new(base: Base, response: Response) -> Result<Data> {
        let answer = response
            .choices
            .into_iter()
            .next()
            .context("Missing Choice")?
            .message
            .content
            .context("Missing answer")?;
        let usage = response.usage.map(Usage::from).unwrap_or_default();

        let mut data = Data {
            base,
            answer,
            answer_tokens: Vec::new(),
            score: SentenceScore {
                counts: Counts::default(),
                reference: 0,
                edits: Vec::new(),
            },
            usage,
        };
        data.rescore();
        Ok(data)
    }

    /// Tokenize the answer and score it again — the only place scoring is applied.
    pub fn rescore(&mut self) {
        self.answer_tokens = scorer::tokenize(&self.answer);
        self.score = python_scorer::score(
            &self.base.tokens,
            &self.answer_tokens,
            &self.base.references,
        );
    }

    /// The model returned the source sentence unchanged.
    fn unchanged(&self) -> bool {
        self.answer_tokens == self.base.tokens
    }

    /// The model reproduced one of the human corrections exactly.
    fn exact_match(&self) -> bool {
        self.base
            .corrections()
            .iter()
            .any(|correction| scorer::tokenize(correction) == self.answer_tokens)
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Usage {
    pub prompt: u64,
    pub result: u64,
    pub total: u64,
}

impl From<crate::openai::Usage> for Usage {
    fn from(usage: crate::openai::Usage) -> Self {
        Self {
            prompt: usage.prompt_tokens,
            result: usage.completion_tokens,
            total: usage.total_tokens,
        }
    }
}

/// A sentence whose request never came back.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Failure {
    pub original: String,
    pub error: String,
}

/// A complete run: what we did, what came back, and what it scored.
#[derive(Debug, Serialize, Deserialize)]
pub struct Run {
    pub provenance: Provenance,
    pub counts: Counts,
    pub metrics: Metrics,
    pub summary: Summary,
    pub failures: Vec<Failure>,
    pub results: Vec<Data>,
}

impl Run {
    pub fn new(provenance: Provenance, results: Vec<Data>, failures: Vec<Failure>) -> Self {
        // Corpus level: accumulate the counts of the reference picked per sentence, then
        // compute the metrics once. Never average per-sentence F0.5.
        let mut counts = Counts::default();
        for data in &results {
            counts.add(data.score.counts);
        }
        let summary = Summary::of(&results);
        Run {
            provenance,
            counts,
            metrics: counts.metrics(),
            summary,
            failures,
            results,
        }
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .wrap_err_with(|| format!("creating {}", parent.display()))?;
        }
        let json = serde_json::to_string_pretty(self).wrap_err("serializing run")?;
        std::fs::write(path, json).wrap_err_with(|| format!("writing {}", path.display()))
    }

    pub fn read(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .wrap_err_with(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&json).wrap_err_with(|| format!("parsing {}", path.display()))
    }

    /// Default location for a run of this model: `results/<model>-<timestamp>.json`.
    pub fn default_output(&self) -> PathBuf {
        let model: String = self
            .provenance
            .model
            .chars()
            .map(
                |c| match c.is_ascii_alphanumeric() || c == '-' || c == '.' {
                    true => c,
                    false => '_',
                },
            )
            .collect();
        PathBuf::from("results").join(format!("{model}-{}.json", self.provenance.started_unix))
    }

    /// The human readable report over the whole run.
    pub fn report(&self) -> String {
        let Provenance {
            tool,
            started,
            model,
            endpoint,
            system,
            prompt,
            dataset,
            ..
        } = &self.provenance;
        let Metrics {
            precision,
            recall,
            f05,
        } = self.metrics;
        let scored = self.results.len();
        let mut report = String::new();
        let _ = write!(report, "\n{model} on {endpoint}\n");
        if let Some(system) = system {
            let _ = writeln!(report, "system    {system:?}");
        }
        let _ = write!(
            report,
            "prompt    {prompt:?}\n\
               data      {} ({} sentences, fnv1a64 {})\n\
               run       {tool}, {started}\n\
             \n\
             scored     {scored} sentences, {} failed\n\
             edits      {} true positive, {} false positive, {} missed\n\
             precision  {precision:.4}\n\
             recall     {recall:.4}\n\
             F0.5       {f05:.4}\n\
             \n\
             exact      {} of {scored} answers equal a human correction\n\
             perfect    {} sentences with no wrong and no missed edit\n\
             unchanged  {} answers left the sentence as it was\n\
             tokens     {} prompt + {} completion\n",
            dataset.path,
            dataset.sentences,
            dataset.hash,
            self.failures.len(),
            self.counts.tp,
            self.counts.fp,
            self.counts.fneg,
            self.summary.exact_match,
            self.summary.perfect,
            self.summary.unchanged,
            self.summary.prompt_tokens,
            self.summary.completion_tokens,
        );
        report
    }
}

/// Descriptive counts that the F0.5 alone hides — a model that never edits anything
/// scores zero precision, and one that rewrites everything scores zero recall.
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct Summary {
    pub sentences: usize,
    pub exact_match: usize,
    pub perfect: usize,
    pub unchanged: usize,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

impl Summary {
    fn of(results: &[Data]) -> Self {
        let mut summary = Summary {
            sentences: results.len(),
            ..Summary::default()
        };
        for data in results {
            summary.exact_match += usize::from(data.exact_match());
            summary.perfect +=
                usize::from(data.score.counts.fp == 0 && data.score.counts.fneg == 0);
            summary.unchanged += usize::from(data.unchanged());
            summary.prompt_tokens += data.usage.prompt;
            summary.completion_tokens += data.usage.result;
        }
        summary
    }
}

/// Everything needed to say what produced a result file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    pub tool: String,
    pub started: String,
    pub started_unix: u64,
    pub model: String,
    pub endpoint: String,
    /// The system prompt, if one was sent; runs written before it existed read as none.
    #[serde(default)]
    pub system: Option<String>,
    pub prompt: String,
    pub dataset: Dataset,
    /// When the stored answers were scored again with a newer scorer.
    pub rescored: Option<String>,
}

impl Provenance {
    pub fn new(
        model: String,
        endpoint: String,
        system: Option<String>,
        prompt: String,
        dataset: Dataset,
    ) -> Self {
        let started_unix = unix_now();
        Provenance {
            tool: format!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")),
            started: utc(started_unix),
            started_unix,
            model,
            endpoint,
            system,
            prompt,
            dataset,
            rescored: None,
        }
    }
}

/// Which gold standard file a run used, hashed so a rerun can be compared against it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub path: String,
    pub bytes: u64,
    /// FNV-1a 64 of the file, as hex — enough to notice that the data changed.
    pub hash: String,
    pub sentences: usize,
}

impl Dataset {
    pub fn of(path: &Path, sentences: usize) -> Result<Self> {
        let bytes = std::fs::read(path).wrap_err_with(|| format!("reading {}", path.display()))?;
        Ok(Dataset {
            path: path.display().to_string(),
            bytes: bytes.len() as u64,
            hash: format!("{:016x}", fnv1a64(&bytes)),
            sentences,
        })
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

/// Format a UNIX timestamp as `YYYY-MM-DDTHH:MM:SSZ`, without pulling in a date crate.
pub fn utc(seconds: u64) -> String {
    let (days, rest) = (seconds / 86_400, seconds % 86_400);
    // Days since 1970-01-01 to a civil date, shifted to a year starting in March so that
    // the leap day falls at the end (Howard Hinnant's civil_from_days).
    let shifted = days as i64 + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let march_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * march_month + 2) / 5 + 1;
    let month = march_month + if march_month < 10 { 3 } else { -9 };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rest / 3_600,
        (rest % 3_600) / 60,
        rest % 60
    )
}

/// What the corpus scores against itself, without spending a cent: a lower bound to
/// beat, an upper bound to aim at, and two checks that the pipeline is sane.
pub fn baselines(corpus: &[Base]) -> String {
    let mut unchanged = Counts::default();
    let mut human = Counts::default();
    let mut annotations = 0;
    let mut retokenized = 0;
    let mut unreachable = Vec::new();

    for base in corpus {
        // A model that answers with the sentence it was given: no edit is ever right.
        unchanged.add(python_scorer::score(&base.tokens, &base.tokens, &base.references).counts);

        for (index, reference) in base.references.iter().enumerate() {
            annotations += 1;

            // One annotator scored the way a model is scored — through the tokenizer, and
            // against every reference including their own. This is the human ceiling.
            let answer = scorer::tokenize(&reference.sentence);
            human.add(python_scorer::score(&base.tokens, &answer, &base.references).counts);

            // The annotation as tokenized by the corpus itself. Where the tokenizer
            // disagrees with it, a model is scored on tokenization, not on grammar.
            let corpus_tokens: Vec<String> = reference
                .sentence
                .split_whitespace()
                .map(str::to_string)
                .collect();
            retokenized += usize::from(answer != corpus_tokens);

            // An annotation scored against nothing but itself should reproduce exactly
            // its own edits. Where it does not, the human edits do not lie on a
            // minimum distance alignment and MaxMatch cannot express them.
            let (counts, _) =
                python_scorer::score_against(&base.tokens, &corpus_tokens, &reference.edits);
            if counts.tp as usize != reference.edits.len() {
                unreachable.push(format!(
                    "  annotator {index} of {:?}: {} of {} edits",
                    base.original,
                    counts.tp,
                    reference.edits.len()
                ));
            }
        }
    }

    let mut report = String::new();
    let line = |report: &mut String, name: &str, counts: Counts| {
        let Metrics {
            precision,
            recall,
            f05,
        } = counts.metrics();
        let _ = writeln!(
            report,
            "{name:<18} P {precision:.4}  R {recall:.4}  F0.5 {f05:.4}   \
             (tp {}, fp {}, fn {})",
            counts.tp, counts.fp, counts.fneg
        );
    };
    let _ = write!(
        report,
        "\n{} sentences, {annotations} annotations, {} gold edits\n\n",
        corpus.len(),
        corpus
            .iter()
            .flat_map(|base| &base.references)
            .map(|reference| reference.edits.len())
            .sum::<usize>(),
    );
    line(&mut report, "answer unchanged", unchanged);
    line(&mut report, "human annotator", human);
    let _ = write!(
        report,
        "\ntokenizer          {retokenized} of {annotations} annotations tokenize differently \
         than the corpus\n\
         maxmatch           {} of {annotations} annotations are not fully expressible as \
         edits on a\n\
         {:19}minimum distance alignment, and can never be scored in full\n",
        unreachable.len(),
        "",
    );
    for annotation in unreachable.iter().take(3) {
        let _ = writeln!(report, "{annotation}");
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::parse_m2;
    use crate::openai::{Choice, Message, Response};

    const SAMPLE: &str = "S 우리는 배가 고펐습니다 .
A 2 3|||R:SPELL|||고팠습니다|||REQUIRED|||-NONE-|||0
A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||1
";

    const LONG_EDIT_SAMPLE: &str = "S a b c d e f g h i
A 0 7|||R:TEST|||x y z q r s t|||REQUIRED|||-NONE-|||0
";

    fn answered(text: &str) -> Data {
        answered_from(SAMPLE, text)
    }

    fn answered_from(sample: &str, text: &str) -> Data {
        let base = parse_m2(sample).unwrap().remove(0);
        let response = Response {
            choices: vec![Choice {
                message: Message {
                    role: "assistant".into(),
                    content: Some(text.into()),
                },
            }],
            usage: None,
        };
        Data::new(base, response).unwrap()
    }

    #[test]
    fn scores_an_answer_end_to_end() {
        // The model wrote the correction of annotator 0, without the space before the
        // full stop that the corpus uses — the tokenizer has to make that a match.
        let corrected = answered("우리는 배가 고팠습니다.");
        assert_eq!(corrected.score.counts.tp, 1);
        assert_eq!(corrected.score.counts.fp, 0);
        assert!(corrected.exact_match());

        // Answering with the sentence as it was matches the `noop` annotator instead.
        let unchanged = answered("우리는 배가 고펐습니다 .");
        assert_eq!(unchanged.score.reference, 1);
        assert_eq!(unchanged.score.counts.tp, 0);
        assert!(unchanged.unchanged());
    }

    #[test]
    fn rescore_uses_the_python_graph_implementation() {
        let data = answered_from(LONG_EDIT_SAMPLE, "x y z q r s t h i");

        assert_eq!(
            (
                data.score.counts.tp,
                data.score.counts.fp,
                data.score.counts.fneg
            ),
            (1, 0, 0)
        );
        assert_eq!(data.score.edits[0].end, 7);
    }

    #[test]
    fn formats_timestamps() {
        assert_eq!(utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(utc(1_772_000_000), "2026-02-25T06:13:20Z");
        // A leap day, to check the March-shifted year arithmetic.
        assert_eq!(utc(1_709_209_845), "2024-02-29T12:30:45Z");
    }

    #[test]
    fn hashes_stably() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_ne!(fnv1a64(b"a"), fnv1a64(b"b"));
    }
}
