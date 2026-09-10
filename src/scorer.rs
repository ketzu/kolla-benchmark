//! MaxMatch (M2) scoring, deterministic and independent of model execution.
//!
//! Given the source tokens, the tokenized model output and the gold references, the
//! scorer searches the decomposition of the model output into edits that maximises the
//! overlap with the gold edits (Dahlmeier & Ng, 2012), including phrase-level edits that
//! merge adjacent operations. It is not a diff: a diff yields *one* edit sequence, while
//! MaxMatch picks the best of all sequences that explain the same output.

use crate::loader::{GoldEdit, Reference};
use serdev::{Deserialize, Serialize};
use std::collections::HashMap;

/// Longest source span a single edit may cover (the corpus' longest gold edit is 6).
const MAX_SOURCE_SPAN: usize = 6;
/// Longest replacement a single edit may cover (the corpus' longest gold one is 22).
const MAX_HYPOTHESIS_SPAN: usize = 24;
/// Unchanged words a phrase-level edit may swallow (m2scorer's `max_unchanged_words`).
const MAX_UNCHANGED_WORDS: usize = 2;
/// Punctuation split off the end of a token, mirroring the KoLLA M2 source tokenization.
const TRAILING_PUNCTUATION: &[char] = &['.', ',', '?', '!', ';', ':'];

/// Split model output the way the KoLLA M2 source lines are tokenized: on whitespace,
/// with sentence-final punctuation as a token of its own. Deliberately no morphological
/// tokenization — that would not line up with the M2 offsets.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for chunk in text.split_whitespace() {
        let head = chunk.trim_end_matches(TRAILING_PUNCTUATION);
        // Keep tokens that are punctuation only (`...`, `^^`) in one piece.
        if head.is_empty() || head.len() == chunk.len() {
            tokens.push(chunk.to_string());
        } else {
            tokens.push(head.to_string());
            tokens.push(chunk[head.len()..].to_string());
        }
    }
    tokens
}

/// True/false positives and false negatives, in edits.
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct Counts {
    pub tp: u64,
    pub fp: u64,
    #[serde(rename = "fn")]
    pub fneg: u64,
}

impl Counts {
    pub fn add(&mut self, other: Counts) {
        self.tp += other.tp;
        self.fp += other.fp;
        self.fneg += other.fneg;
    }

    /// Corpus-level precision, recall and F0.5. Undefined ratios are reported as zero.
    pub fn metrics(&self) -> Metrics {
        let precision = ratio(self.tp, self.tp + self.fp);
        let recall = ratio(self.tp, self.tp + self.fneg);
        let f05 = match 0.25 * precision + recall {
            denominator if denominator > 0.0 => 1.25 * precision * recall / denominator,
            _ => 0.0,
        };
        Metrics {
            precision,
            recall,
            f05,
        }
    }

    /// F0.5 used only to pick the best reference for a sentence. A sentence where
    /// nothing had to be corrected and nothing was corrected counts as perfect, so that
    /// such a reference wins over one that demands edits.
    fn sentence_f05(&self) -> f64 {
        match self.tp + self.fp + self.fneg {
            0 => 1.0,
            _ => self.metrics().f05,
        }
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    match denominator {
        0 => 0.0,
        _ => numerator as f64 / denominator as f64,
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct Metrics {
    pub precision: f64,
    pub recall: f64,
    pub f05: f64,
}

/// An edit the model made: source tokens `start..end` became `replacement`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemEdit {
    pub start: usize,
    pub end: usize,
    pub replacement: Vec<String>,
}

/// The score of one model output, against the reference that suited it best.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceScore {
    pub counts: Counts,
    /// Index into the sentence's references — which annotator was scored against.
    pub reference: usize,
    /// The edit decomposition MaxMatch chose for that reference.
    pub edits: Vec<SystemEdit>,
}

/// Score one model output against every reference and keep the best (point 6 of the
/// procedure: per sentence best F0.5, never an average of per-sentence scores).
pub fn score(source: &[String], hypothesis: &[String], references: &[Reference]) -> SentenceScore {
    if references.is_empty() {
        let (counts, edits) = score_against(source, hypothesis, &[]);
        return SentenceScore {
            counts,
            reference: 0,
            edits,
        };
    }
    references
        .iter()
        .enumerate()
        .map(|(reference, gold)| {
            let (counts, edits) = score_against(source, hypothesis, &gold.edits);
            SentenceScore {
                counts,
                reference,
                edits,
            }
        })
        // Best F0.5 wins; ties go to more matches, then fewer mistakes, then the first
        // reference, so the choice never depends on iteration order.
        .max_by(|a, b| {
            let errors = |s: &SentenceScore| s.counts.fp + s.counts.fneg;
            a.counts
                .sentence_f05()
                .total_cmp(&b.counts.sentence_f05())
                .then(a.counts.tp.cmp(&b.counts.tp))
                .then(errors(b).cmp(&errors(a)))
                .then(b.reference.cmp(&a.reference))
        })
        .expect("references is not empty")
}

/// MaxMatch against a single reference: the edit decomposition maximising true positives,
/// and among those the one with the fewest edits (i.e. the fewest false positives).
pub fn score_against(
    source: &[String],
    hypothesis: &[String],
    gold: &[GoldEdit],
) -> (Counts, Vec<SystemEdit>) {
    let mut accepted: HashMap<(usize, usize), Vec<Vec<String>>> = HashMap::new();
    for edit in gold {
        accepted
            .entry((edit.start, edit.end))
            .or_default()
            .extend(edit.replacements.iter().map(|r| tokenize(r)));
    }

    let (rows, columns) = (source.len() + 1, hypothesis.len() + 1);
    // Only alignments of minimal edit distance are considered, as in m2scorer: without
    // that, a model could be credited for an edit it never made, by "inserting" a word
    // that is already there and deleting the original one.
    let alignment = Alignment::of(source, hypothesis);
    let mut cells = Lattice::new(rows, columns);
    cells.set(Vertex::start(), Cell::default());

    // Vertices are visited in row-major order and every arc moves forward, so a vertex is
    // final by the time it is expanded.
    for i in 0..rows {
        for j in 0..columns {
            for inserted_here in [false, true] {
                let at = Vertex {
                    source: i,
                    hypothesis: j,
                    inserted_here,
                };
                let Some(cell) = cells.get(at).cloned() else {
                    continue;
                };
                // Carry an unchanged token over, free of charge.
                if i + 1 < rows
                    && j + 1 < columns
                    && source[i] == hypothesis[j]
                    && alignment.arc(i, j, i + 1, j + 1, 0)
                {
                    cells.relax(at.step(i + 1, j + 1), &cell, at, None);
                }
                for end in i..rows.min(i + MAX_SOURCE_SPAN + 1) {
                    // One insertion per source position: two of them could otherwise both
                    // match the same gold insertion and count it twice. Two gold
                    // insertions at one position can therefore only score one match.
                    if end == i && inserted_here {
                        continue;
                    }
                    for stop in j..columns.min(j + MAX_HYPOTHESIS_SPAN + 1) {
                        let (removed, added) = (&source[i..end], &hypothesis[j..stop]);
                        if removed == added {
                            continue; // Not an edit — either empty or unchanged.
                        }
                        // Edits are tight: a merged edit never starts or ends on an
                        // unchanged token.
                        if !removed.is_empty()
                            && !added.is_empty()
                            && (removed[0] == added[0]
                                || removed[removed.len() - 1] == added[added.len() - 1])
                        {
                            continue;
                        }
                        let cost = distance(removed, added);
                        if !alignment.arc(i, j, end, stop, cost) {
                            continue;
                        }
                        // A merged edit swallows at most MAX_UNCHANGED_WORDS unchanged
                        // words: aligning p against q words at cost c leaves max(p, q) - c
                        // of them untouched.
                        if removed.len().max(added.len()) - cost as usize > MAX_UNCHANGED_WORDS {
                            continue;
                        }
                        let edit = SystemEdit {
                            start: i,
                            end,
                            replacement: added.to_vec(),
                        };
                        let correct = accepted
                            .get(&(i, end))
                            .is_some_and(|options| options.iter().any(|o| o == added));
                        let to = Vertex {
                            source: end,
                            hypothesis: stop,
                            inserted_here: end == i,
                        };
                        cells.relax(to, &cell, at, Some((edit, correct)));
                    }
                }
            }
        }
    }

    let end = cells
        .best_end()
        .expect("every hypothesis is reachable by single token edits");
    let best = cells.get(end).expect("just found");
    let counts = Counts {
        tp: best.tp as u64,
        fp: (best.edits - best.tp) as u64,
        fneg: (gold.len() as u64).saturating_sub(best.tp as u64),
    };
    (counts, cells.backtrack(end))
}

/// A position in the search: how much of the source and of the hypothesis is consumed,
/// and whether the last arc was an insertion that left the source position where it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Vertex {
    source: usize,
    hypothesis: usize,
    inserted_here: bool,
}

impl Vertex {
    fn start() -> Self {
        Vertex {
            source: 0,
            hypothesis: 0,
            inserted_here: false,
        }
    }

    fn step(self, source: usize, hypothesis: usize) -> Self {
        Vertex {
            source,
            hypothesis,
            inserted_here: false,
        }
    }
}

#[derive(Debug, Default, Clone)]
struct Cell {
    tp: u32,
    edits: u32,
    from: Option<Vertex>,
    edit: Option<SystemEdit>,
}

/// The search space: the best way found so far to reach each vertex.
struct Lattice {
    cells: Vec<Option<Cell>>,
    columns: usize,
}

impl Lattice {
    fn new(rows: usize, columns: usize) -> Self {
        Lattice {
            cells: vec![None; rows * columns * 2],
            columns,
        }
    }

    fn index(&self, at: Vertex) -> usize {
        (at.source * self.columns + at.hypothesis) * 2 + usize::from(at.inserted_here)
    }

    fn get(&self, at: Vertex) -> Option<&Cell> {
        self.cells[self.index(at)].as_ref()
    }

    fn set(&mut self, at: Vertex, cell: Cell) {
        let index = self.index(at);
        self.cells[index] = Some(cell);
    }

    /// Keep the way to `at` with the most matches, and among those the fewest edits.
    fn relax(
        &mut self,
        at: Vertex,
        previous: &Cell,
        from: Vertex,
        edit: Option<(SystemEdit, bool)>,
    ) {
        let (edit, correct) = match edit {
            Some((edit, correct)) => (Some(edit), correct),
            None => (None, false),
        };
        let candidate = Cell {
            tp: previous.tp + u32::from(correct),
            edits: previous.edits + u32::from(edit.is_some()),
            from: Some(from),
            edit,
        };
        let better = match self.get(at) {
            None => true,
            Some(current) => {
                candidate.tp > current.tp
                    || (candidate.tp == current.tp && candidate.edits < current.edits)
            }
        };
        if better {
            self.set(at, candidate);
        }
    }

    /// The better of the two ways to have consumed everything.
    fn best_end(&self) -> Option<Vertex> {
        let end = |inserted_here| Vertex {
            source: self.cells.len() / (self.columns * 2) - 1,
            hypothesis: self.columns - 1,
            inserted_here,
        };
        match (self.get(end(false)), self.get(end(true))) {
            (Some(plain), Some(inserted)) => match inserted.tp > plain.tp
                || (inserted.tp == plain.tp && inserted.edits < plain.edits)
            {
                true => Some(end(true)),
                false => Some(end(false)),
            },
            (Some(_), None) => Some(end(false)),
            (None, Some(_)) => Some(end(true)),
            (None, None) => None,
        }
    }

    fn backtrack(&self, end: Vertex) -> Vec<SystemEdit> {
        let mut edits = Vec::new();
        let mut at = Some(end);
        while let Some(vertex) = at {
            let Some(cell) = self.get(vertex) else {
                break;
            };
            if let Some(edit) = &cell.edit {
                edits.push(edit.clone());
            }
            at = cell.from;
        }
        edits.reverse();
        edits
    }
}

/// Levenshtein distance of one edit block, with unit costs.
fn distance(left: &[String], right: &[String]) -> u32 {
    let mut previous: Vec<u32> = (0..=right.len() as u32).collect();
    let mut current = vec![0u32; right.len() + 1];
    for (row, l) in left.iter().enumerate() {
        current[0] = row as u32 + 1;
        for (column, r) in right.iter().enumerate() {
            let substitute = previous[column] + u32::from(l != r);
            current[column + 1] = substitute
                .min(previous[column + 1] + 1)
                .min(current[column] + 1);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()]
}

/// Which vertices and arcs lie on an alignment of minimal edit distance.
struct Alignment {
    /// Distance from the start of both sequences to each vertex.
    forward: Vec<u32>,
    /// Distance from each vertex to the end of both sequences.
    backward: Vec<u32>,
    columns: usize,
    best: u32,
}

impl Alignment {
    fn of(source: &[String], hypothesis: &[String]) -> Self {
        let (rows, columns) = (source.len() + 1, hypothesis.len() + 1);
        let mut forward = vec![0u32; rows * columns];
        let mut backward = vec![0u32; rows * columns];
        for i in 0..rows {
            for j in 0..columns {
                forward[i * columns + j] = match (i, j) {
                    (0, _) => j as u32,
                    (_, 0) => i as u32,
                    _ => (forward[(i - 1) * columns + j - 1]
                        + u32::from(source[i - 1] != hypothesis[j - 1]))
                    .min(forward[(i - 1) * columns + j] + 1)
                    .min(forward[i * columns + j - 1] + 1),
                };
            }
        }
        for i in (0..rows).rev() {
            for j in (0..columns).rev() {
                backward[i * columns + j] = match (rows - 1 - i, columns - 1 - j) {
                    (0, remaining) => remaining as u32,
                    (remaining, 0) => remaining as u32,
                    _ => (backward[(i + 1) * columns + j + 1]
                        + u32::from(source[i] != hypothesis[j]))
                    .min(backward[(i + 1) * columns + j] + 1)
                    .min(backward[i * columns + j + 1] + 1),
                };
            }
        }
        let best = forward[rows * columns - 1];
        Alignment {
            forward,
            backward,
            columns,
            best,
        }
    }

    /// True when going from one vertex to another at `cost` stays on a minimal alignment.
    fn arc(&self, i: usize, j: usize, end: usize, stop: usize, cost: u32) -> bool {
        let (from, to) = (i * self.columns + j, end * self.columns + stop);
        self.forward[from] + self.backward[from] == self.best
            && self.forward[to] + self.backward[to] == self.best
            && self.forward[to] == self.forward[from] + cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_string).collect()
    }

    fn reference(annotator: u8, edits: &[(usize, usize, &str)]) -> Reference {
        Reference {
            annotator,
            sentence: String::new(),
            edits: edits
                .iter()
                .map(|(start, end, replacement)| GoldEdit {
                    start: *start,
                    end: *end,
                    replacements: vec![replacement.to_string()],
                })
                .collect(),
        }
    }

    #[test]
    fn tokenizes_like_the_m2_source_lines() {
        assert_eq!(
            tokenize("우리는 배가 고팠습니다."),
            tokens("우리는 배가 고팠습니다 .")
        );
        assert_eq!(tokenize(" a  b \n c "), tokens("a b c"));
        // Punctuation-only tokens stay whole, inner punctuation is never split.
        assert_eq!(tokenize("... 10,000달러인데"), tokens("... 10,000달러인데"));
    }

    #[test]
    fn finds_a_matching_edit() {
        let gold = [reference(0, &[(2, 3, "고팠습니다")])];
        let score = score(
            &tokens("우리는 배가 고펐습니다 ."),
            &tokens("우리는 배가 고팠습니다 ."),
            &gold,
        );
        assert_eq!(
            (score.counts.tp, score.counts.fp, score.counts.fneg),
            (1, 0, 0)
        );
        assert_eq!(score.edits.len(), 1);
        assert_eq!(score.edits[0].replacement, tokens("고팠습니다"));
    }

    #[test]
    fn counts_missed_and_spurious_edits() {
        let gold = [reference(0, &[(2, 3, "고팠습니다")])];
        // Model changed nothing: the gold edit is a false negative.
        let missed = score(&tokens("a b c"), &tokens("a b c"), &gold);
        assert_eq!(
            (missed.counts.tp, missed.counts.fp, missed.counts.fneg),
            (0, 0, 1)
        );
        // Model changed the wrong token: one false positive on top of the miss.
        let spurious = score(&tokens("a b c"), &tokens("a x c"), &gold);
        assert_eq!(
            (spurious.counts.tp, spurious.counts.fp, spurious.counts.fneg),
            (0, 1, 1)
        );
        // A `noop` reference has no gold edits at all.
        let noop = score(&tokens("a b c"), &tokens("a x c"), &[reference(0, &[])]);
        assert_eq!(
            (noop.counts.tp, noop.counts.fp, noop.counts.fneg),
            (0, 1, 0)
        );
    }

    #[test]
    fn merges_adjacent_operations_into_a_phrase_edit() {
        // The two token changes must be merged to match the single gold edit …
        let phrase = score(
            &tokens("a b c"),
            &tokens("x y c"),
            &[reference(0, &[(0, 2, "x y")])],
        );
        assert_eq!((phrase.counts.tp, phrase.counts.fp), (1, 0));
        // … and must stay separate when the gold standard has them separate.
        let split = score(
            &tokens("a b c"),
            &tokens("x y c"),
            &[reference(0, &[(0, 1, "x"), (1, 2, "y")])],
        );
        assert_eq!((split.counts.tp, split.counts.fp), (2, 0));
    }

    #[test]
    fn never_credits_an_edit_the_model_did_not_make() {
        // The gold standard wants a second "b" inserted. The model answered with the
        // sentence unchanged — pretending it inserted the "b" that is already there and
        // deleted the original one would explain the same output, but is not an edit the
        // model made, so only the alignment of minimal distance counts: no edit at all.
        let gold = [reference(0, &[(1, 1, "b")])];
        let score = score(&tokens("a b c"), &tokens("a b c"), &gold);
        assert_eq!(
            (score.counts.tp, score.counts.fp, score.counts.fneg),
            (0, 0, 1)
        );
        assert!(score.edits.is_empty());
    }

    #[test]
    fn scores_at_most_one_insertion_per_position() {
        // Two words inserted one after the other are two gold edits at 0..0. Only one
        // insertion per source position is allowed, which keeps a single gold insertion
        // from being counted twice — at the price of these two being merged into one
        // edit that matches neither. One gold insertion is matched as usual.
        let gold = [reference(0, &[(0, 0, "x"), (0, 0, "y")])];
        let merged = score(&tokens("a b"), &tokens("x y a b"), &gold);
        assert_eq!(
            (merged.counts.tp, merged.counts.fp, merged.counts.fneg),
            (0, 1, 2)
        );

        let single = score(
            &tokens("a b"),
            &tokens("x a b"),
            &[reference(0, &[(0, 0, "x")])],
        );
        assert_eq!((single.counts.tp, single.counts.fp), (1, 0));
    }

    #[test]
    fn picks_the_reference_with_the_best_f05() {
        let references = [
            reference(0, &[(1, 2, "x"), (2, 3, "y")]),
            reference(1, &[(1, 2, "x")]),
        ];
        // The model made exactly the edit of annotator 1: perfect against it, half a
        // recall miss against annotator 0.
        let score = score(&tokens("a b c"), &tokens("a x c"), &references);
        assert_eq!(score.reference, 1);
        assert_eq!(
            (score.counts.tp, score.counts.fp, score.counts.fneg),
            (1, 0, 0)
        );
    }

    #[test]
    fn computes_corpus_metrics() {
        let mut counts = Counts::default();
        counts.add(Counts {
            tp: 2,
            fp: 1,
            fneg: 3,
        });
        let metrics = counts.metrics();
        assert!((metrics.precision - 2.0 / 3.0).abs() < 1e-9);
        assert!((metrics.recall - 0.4).abs() < 1e-9);
        assert!((metrics.f05 - 0.588_235_294).abs() < 1e-6);
        assert_eq!(Counts::default().metrics().f05, 0.0);
    }
}
