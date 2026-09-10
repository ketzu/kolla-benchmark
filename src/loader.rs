use eyre::{Context, Result, eyre};
use serdev::{Deserialize, Serialize};
use std::path::Path;

/// One benchmark item: the erroneous learner sentence plus every human reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Base {
    /// Source sentence, re-joined from the M2 source tokens.
    pub original: String,
    /// The M2 source tokenization; all gold edit offsets index into this.
    pub tokens: Vec<String>,
    /// One reference annotation set per annotator.
    pub references: Vec<Reference>,
}

impl Base {
    /// The corrected sentences, one per annotator.
    pub fn corrections(&self) -> Vec<String> {
        self.references.iter().map(|r| r.sentence.clone()).collect()
    }

    /// Build one sentence from its source tokens and the edits of all annotators.
    fn from_m2(tokens: &[&str], edits: &[Edit]) -> Result<Self> {
        let mut annotators: Vec<u8> = edits.iter().map(|edit| edit.annotator).collect();
        annotators.sort_unstable();
        annotators.dedup();

        let references = annotators
            .iter()
            .map(|annotator| {
                Reference::from_m2(tokens, edits, *annotator)
                    .wrap_err_with(|| format!("annotator {annotator}"))
            })
            .collect::<Result<_>>()?;

        Ok(Self {
            original: tokens.join(" "),
            tokens: tokens.iter().map(|t| (*t).to_string()).collect(),
            references,
        })
    }
}

/// One annotator's gold standard for a sentence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reference {
    pub annotator: u8,
    /// The source sentence with this annotator's edits applied.
    pub sentence: String,
    /// Gold edits, sorted by start and non-overlapping; a `noop` annotation yields none.
    pub edits: Vec<GoldEdit>,
}

impl Reference {
    fn from_m2(tokens: &[&str], edits: &[Edit], annotator: u8) -> Result<Self> {
        let mut own: Vec<&Edit> = edits.iter().filter(|e| e.annotator == annotator).collect();
        own.sort_by_key(|e| e.start);
        let sentence = apply(tokens, &own)?;

        let mut gold: Vec<GoldEdit> = Vec::new();
        for edit in own {
            // `noop`, and any edit replacing nothing with nothing, is not a correction.
            if edit.start == edit.end && edit.replacement.split_whitespace().count() == 0 {
                continue;
            }
            gold.push(GoldEdit {
                start: edit.start,
                end: edit.end,
                // Each `A` line is one correction of its own — two lines with the same
                // span are two edits, e.g. two words inserted one after the other. Only
                // `||` inside the replacement marks accepted alternatives.
                replacements: edit
                    .replacement
                    .split("||")
                    .map(|replacement| replacement.trim().to_string())
                    .collect(),
            });
        }

        Ok(Self {
            annotator,
            sentence,
            edits: gold,
        })
    }
}

/// A gold correction: source tokens `start..end` may be replaced by any of `replacements`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldEdit {
    pub start: usize,
    pub end: usize,
    pub replacements: Vec<String>,
}

/// A single `A` line: replace `start..end` of the source tokens with `replacement`.
/// A `noop` line becomes an empty edit (`0..0` replaced by nothing), so the
/// annotator still contributes an unchanged reference sentence.
#[derive(Debug)]
struct Edit {
    start: usize,
    end: usize,
    replacement: String,
    annotator: u8,
}

impl Edit {
    /// Parse an `A` line body: `start end|||type|||replacement|||…|||annotator`.
    fn parse(body: &str, tokens: usize) -> Result<Self> {
        let fields: Vec<&str> = body.split("|||").collect();
        let [offsets, _kind, replacement, .., annotator] = fields.as_slice() else {
            return Err(eyre!("expected at least 4 `|||` separated fields"));
        };
        let (start, end) = offsets
            .split_once(' ')
            .ok_or_else(|| eyre!("malformed offsets {offsets:?}"))?;
        let (start, end): (i64, i64) = (start.trim().parse()?, end.trim().parse()?);
        let annotator = annotator.trim().parse()?;

        if start < 0 {
            return Ok(Self {
                start: 0,
                end: 0,
                replacement: String::new(),
                annotator,
            });
        }
        let (start, end) = (start as usize, end as usize);
        if start > end || end > tokens {
            return Err(eyre!(
                "edit {start}..{end} outside of {tokens} source tokens"
            ));
        }
        Ok(Self {
            start,
            end,
            replacement: replacement.to_string(),
            annotator,
        })
    }
}

/// Splice non-overlapping, start-sorted edits into the source tokens.
fn apply(tokens: &[&str], edits: &[&Edit]) -> Result<String> {
    let mut out: Vec<&str> = Vec::new();
    let mut next = 0;
    for edit in edits {
        if edit.start < next {
            return Err(eyre!("edit at {} overlaps the previous one", edit.start));
        }
        out.extend(&tokens[next..edit.start]);
        // Of several accepted alternatives, the first one builds the reference sentence.
        let replacement = edit.replacement.split("||").next().unwrap_or_default();
        out.extend(replacement.split_whitespace());
        next = edit.end;
    }
    out.extend(&tokens[next..]);
    Ok(out.join(" "))
}

/// Parse an M2 file: `S` source lines, each followed by the annotators' `A` edit lines.
pub fn parse_m2(input: &str) -> Result<Vec<Base>> {
    let mut sentences = Vec::new();
    let mut tokens: Option<Vec<&str>> = None;
    let mut edits: Vec<Edit> = Vec::new();

    for (index, line) in input.lines().enumerate() {
        let number = index + 1;

        if let Some(source) = line.strip_prefix("S ") {
            if let Some(previous) = tokens.take() {
                sentences.push(
                    Base::from_m2(&previous, &edits)
                        .wrap_err_with(|| format!("sentence ending at line {number}"))?,
                );
            }
            edits.clear();
            tokens = Some(source.split_whitespace().collect());
        } else if let Some(body) = line.strip_prefix("A ") {
            let tokens = tokens
                .as_ref()
                .ok_or_else(|| eyre!("line {number}: edit before any source sentence"))?;
            edits.push(Edit::parse(body, tokens.len()).wrap_err_with(|| format!("line {number}"))?);
        } else if !line.trim().is_empty() {
            return Err(eyre!("line {number}: expected `S `, `A ` or a blank line"));
        }
    }
    if let Some(last) = tokens {
        sentences.push(Base::from_m2(&last, &edits)?);
    }
    Ok(sentences)
}

/// Read and parse an M2 file from disk.
pub fn load_m2(path: impl AsRef<Path>) -> Result<Vec<Base>> {
    let path = path.as_ref();
    let input =
        std::fs::read_to_string(path).wrap_err_with(|| format!("reading {}", path.display()))?;
    parse_m2(&input).wrap_err_with(|| format!("parsing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
S 그날에 우리는 한국으로 출발했습니다 .
A 0 1|||R:ADV+ADP -> NOUN|||그날|||REQUIRED|||-NONE-|||0
A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||1

S 서울에서 아침을 울 같이 아짐을 먹고 .
A 4 5|||R:SPELL|||아침을|||REQUIRED|||-NONE-|||0
A 2 3|||R:SPELL|||우리|||REQUIRED|||-NONE-|||1
A 4 5|||U:NOUN||||||REQUIRED|||-NONE-|||1

S 저녁 비행기를 출발했습니다 .
A 1 1|||M:NUM ADV|||열 시에|||REQUIRED|||-NONE-|||0
";

    #[test]
    fn parses_sentences_and_annotators() {
        let sentences = parse_m2(SAMPLE).unwrap();
        assert_eq!(sentences.len(), 3);

        let first = &sentences[0];
        assert_eq!(first.original, "그날에 우리는 한국으로 출발했습니다 .");
        assert_eq!(first.tokens.len(), 5);
        // Replacement for annotator 0, unchanged sentence for the `noop` annotator 1.
        assert_eq!(
            first.corrections(),
            vec![
                "그날 우리는 한국으로 출발했습니다 .",
                "그날에 우리는 한국으로 출발했습니다 ."
            ]
        );
        let gold = &first.references[0].edits;
        assert_eq!(gold.len(), 1);
        assert_eq!((gold[0].start, gold[0].end), (0, 1));
        // A `noop` reference has no gold edits, so there is nothing to find.
        assert!(first.references[1].edits.is_empty());
    }

    #[test]
    fn applies_multiple_edits_deletions_and_insertions() {
        let sentences = parse_m2(SAMPLE).unwrap();
        assert_eq!(
            sentences[1].corrections(),
            vec![
                "서울에서 아침을 울 같이 아침을 먹고 .",
                // Two edits of annotator 1: replacement plus a deletion.
                "서울에서 아침을 우리 같이 먹고 .",
            ]
        );
        // The deletion survives as a gold edit with an empty replacement.
        let deletion = &sentences[1].references[1].edits[1];
        assert_eq!(sentences[1].references[1].edits.len(), 2);
        assert_eq!((deletion.start, deletion.end), (4, 5));
        assert_eq!(deletion.replacements, vec![""]);
        // Insertion of two tokens, single annotator, last sentence without trailing blank line.
        assert_eq!(
            sentences[2].corrections(),
            vec!["저녁 열 시에 비행기를 출발했습니다 ."]
        );
    }

    #[test]
    fn keeps_two_edits_on_the_same_span_apart() {
        // Two `A` lines for the same span insert two words in sequence — they are not
        // alternatives for one another.
        let sentences = parse_m2(
            "S b c\n\
             A 0 0|||M:PRON|||x|||REQUIRED|||-NONE-|||0\n\
             A 0 0|||M:SCONJ|||y|||REQUIRED|||-NONE-|||0\n",
        )
        .unwrap();
        assert_eq!(sentences[0].corrections(), vec!["x y b c"]);
        assert_eq!(sentences[0].references[0].edits.len(), 2);
    }

    #[test]
    fn rejects_malformed_input() {
        assert!(parse_m2("A 0 1|||R:SPELL|||x|||REQUIRED|||-NONE-|||0\n").is_err());
        assert!(parse_m2("S a b\nA 0 9|||R:SPELL|||x|||REQUIRED|||-NONE-|||0\n").is_err());
        assert!(parse_m2("S a b\nnonsense\n").is_err());
    }
}
