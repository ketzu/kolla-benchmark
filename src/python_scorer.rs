//! MaxMatch scoring through the graph construction used by the Python m2scorer.

use crate::loader::{GoldEdit, Reference};
use crate::scorer::{Counts, SentenceScore, SystemEdit};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

const MAX_UNCHANGED_WORDS: usize = 2;
const EPSILON: f64 = 0.001;

type Vertex = (usize, usize);
type Span = (usize, usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EditKind {
    Ins,
    Del,
    Sub,
    Noop,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GraphEdit {
    kind: EditKind,
    start: usize,
    end: usize,
    original: String,
    correction: String,
    unchanged: usize,
}

impl GraphEdit {
    fn is_noop(&self) -> bool {
        self.kind == EditKind::Noop
    }
}

#[derive(Debug, Clone)]
struct Edge {
    from: Vertex,
    to: Vertex,
    edit: GraphEdit,
    weight: f64,
}

/// A directed edit graph. The edge index is the Rust equivalent of Python's `edits`
/// dictionary, while the vector preserves the deterministic edge traversal order.
#[derive(Debug, Clone, Default)]
struct Graph {
    vertices: Vec<Vertex>,
    edges: Vec<Edge>,
    edge_index: HashMap<(Vertex, Vertex), usize>,
}

impl Graph {
    fn add_edge(&mut self, from: Vertex, to: Vertex, edit: GraphEdit, weight: f64) {
        if let Some(&index) = self.edge_index.get(&(from, to)) {
            if weight < self.edges[index].weight {
                self.edges[index].weight = weight;
                self.edges[index].edit = edit;
            }
            return;
        }
        let index = self.edges.len();
        self.edges.push(Edge {
            from,
            to,
            edit,
            weight,
        });
        self.edge_index.insert((from, to), index);
    }

    fn rebuild_index(&mut self) {
        self.edge_index.clear();
        for (index, edge) in self.edges.iter().enumerate() {
            self.edge_index.insert((edge.from, edge.to), index);
        }
    }

    fn sort_initial(&mut self) {
        self.vertices.sort_unstable();
        self.vertices.dedup();
        self.edges.sort_unstable_by_key(|edge| (edge.from, edge.to));
        self.rebuild_index();
    }

    fn edge(&self, from: Vertex, to: Vertex) -> Option<&Edge> {
        self.edge_index
            .get(&(from, to))
            .map(|&index| &self.edges[index])
    }
}

#[derive(Debug, Clone)]
struct GoldView {
    start: usize,
    end: usize,
    original: String,
    replacements: Vec<String>,
}

/// Score one model output against every reference and keep the best F0.5 reference.
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
        .max_by(|a, b| {
            let errors = |sentence: &SentenceScore| sentence.counts.fp + sentence.counts.fneg;
            sentence_f05(&a.counts)
                .total_cmp(&sentence_f05(&b.counts))
                .then(a.counts.tp.cmp(&b.counts.tp))
                .then(errors(b).cmp(&errors(a)))
                .then(b.reference.cmp(&a.reference))
        })
        .expect("references is not empty")
}

/// Score against one reference using the Python scorer's two-graph algorithm.
pub fn score_against(
    source: &[String],
    hypothesis: &[String],
    gold: &[GoldEdit],
) -> (Counts, Vec<SystemEdit>) {
    let first = levenshtein_graph(source, hypothesis, 1);
    let second = levenshtein_graph(source, hypothesis, 2);
    let mut graph = merge_graph(first, second);
    add_transitive_arcs(&mut graph);
    set_weights(&mut graph, source, gold);

    // Python backtracks from the endpoint and then reverses the sequence when matching
    // it. Keep the public Rust result in forward order, like scorer::score_against.
    let reverse_sequence = best_edit_sequence(&graph);
    let sequence: Vec<GraphEdit> = reverse_sequence.into_iter().rev().collect();
    let gold_views = gold_views(source, gold);
    let true_positives = matching_count(&sequence, &gold_views);
    let edits = sequence
        .iter()
        .filter(|edit| !edit.is_noop())
        .map(system_edit)
        .collect::<Vec<_>>();

    let counts = Counts {
        tp: true_positives as u64,
        fp: (edits.len() - true_positives) as u64,
        fneg: (gold.len() - true_positives) as u64,
    };
    (counts, edits)
}

fn sentence_f05(counts: &Counts) -> f64 {
    match counts.tp + counts.fp + counts.fneg {
        0 => 1.0,
        _ => counts.metrics().f05,
    }
}

fn system_edit(edit: &GraphEdit) -> SystemEdit {
    SystemEdit {
        start: edit.start,
        end: edit.end,
        replacement: edit
            .correction
            .split_whitespace()
            .map(str::to_string)
            .collect(),
    }
}

fn gold_views(source: &[String], gold: &[GoldEdit]) -> Vec<GoldView> {
    gold.iter()
        .map(|edit| GoldView {
            start: edit.start,
            end: edit.end,
            original: source[edit.start..edit.end].join(" "),
            replacements: edit.replacements.clone(),
        })
        .collect()
}

fn matches_gold(edit: &GraphEdit, gold: &GoldView) -> bool {
    edit.start == gold.start
        && edit.end == gold.end
        && edit.original == gold.original
        && gold
            .replacements
            .iter()
            .any(|replacement| replacement == &edit.correction)
}

fn matching_count(sequence: &[GraphEdit], gold: &[GoldView]) -> usize {
    let mut matches = 0;
    let mut last_gold = 0;
    for edit in sequence {
        if let Some(index) = (last_gold..gold.len()).find(|&index| matches_gold(edit, &gold[index]))
        {
            matches += 1;
            last_gold = index + 1;
        }
    }
    matches
}

fn levenshtein_graph(first: &[String], second: &[String], substitution_cost: usize) -> Graph {
    let rows = first.len() + 1;
    let columns = second.len() + 1;
    let mut distances = vec![vec![0usize; columns]; rows];
    let mut backpointers: Vec<Vec<Vec<(Vertex, GraphEdit)>>> =
        vec![vec![Vec::new(); columns]; rows];

    for i in 1..rows {
        distances[i][0] = i;
        backpointers[i][0].push((
            (i - 1, 0),
            GraphEdit {
                kind: EditKind::Del,
                start: i - 1,
                end: i,
                original: first[i - 1].clone(),
                correction: String::new(),
                unchanged: 0,
            },
        ));
    }
    for j in 1..columns {
        distances[0][j] = j;
        backpointers[0][j].push((
            (0, j - 1),
            GraphEdit {
                kind: EditKind::Ins,
                start: 0,
                end: 0,
                original: String::new(),
                correction: second[j - 1].clone(),
                unchanged: 0,
            },
        ));
    }

    for i in 1..rows {
        for j in 1..columns {
            let substitution = distances[i - 1][j - 1]
                + usize::from(first[i - 1] != second[j - 1]) * substitution_cost;
            let deletion = distances[i - 1][j] + 1;
            let insertion = distances[i][j - 1] + 1;
            let minimum = substitution.min(deletion).min(insertion);

            if substitution == minimum {
                distances[i][j] = minimum;
                backpointers[i][j].push((
                    (i - 1, j - 1),
                    if first[i - 1] == second[j - 1] {
                        GraphEdit {
                            kind: EditKind::Noop,
                            start: i - 1,
                            end: i,
                            original: first[i - 1].clone(),
                            correction: second[j - 1].clone(),
                            unchanged: 1,
                        }
                    } else {
                        GraphEdit {
                            kind: EditKind::Sub,
                            start: i - 1,
                            end: i,
                            original: first[i - 1].clone(),
                            correction: second[j - 1].clone(),
                            unchanged: 0,
                        }
                    },
                ));
            }
            if deletion == minimum {
                distances[i][j] = minimum;
                backpointers[i][j].push((
                    (i - 1, j),
                    GraphEdit {
                        kind: EditKind::Del,
                        start: i - 1,
                        end: i,
                        original: first[i - 1].clone(),
                        correction: String::new(),
                        unchanged: 0,
                    },
                ));
            }
            if insertion == minimum {
                distances[i][j] = minimum;
                backpointers[i][j].push((
                    (i, j - 1),
                    GraphEdit {
                        kind: EditKind::Ins,
                        start: i,
                        end: i,
                        original: String::new(),
                        correction: second[j - 1].clone(),
                        unchanged: 0,
                    },
                ));
            }
        }
    }

    let mut graph = Graph::default();
    let endpoint = (first.len(), second.len());
    let mut queue = VecDeque::from([endpoint]);
    let mut visited = HashSet::new();
    while let Some(vertex) = queue.pop_front() {
        if !visited.insert(vertex) {
            continue;
        }
        graph.vertices.push(vertex);
        for (previous, edit) in &backpointers[vertex.0][vertex.1] {
            graph.add_edge(*previous, vertex, edit.clone(), 1.0);
            queue.push_back(*previous);
        }
    }
    graph.sort_initial();
    graph
}

fn merge_graph(first: Graph, second: Graph) -> Graph {
    let mut graph = first;
    graph.vertices.extend(second.vertices);
    for edge in second.edges {
        graph.add_edge(edge.from, edge.to, edge.edit, edge.weight);
    }
    graph.sort_initial();
    graph
}

fn merge_edits(first: &GraphEdit, second: &GraphEdit) -> GraphEdit {
    use EditKind::{Del, Ins, Noop, Sub};

    let (kind, original, correction) = match (first.kind, second.kind) {
        (Ins, Ins) => (
            Ins,
            String::new(),
            join(&first.correction, &second.correction),
        ),
        (Ins, Del) => (Sub, second.original.clone(), first.correction.clone()),
        (Ins, Sub | Noop) => (
            Sub,
            second.original.clone(),
            join(&first.correction, &second.correction),
        ),
        (Del, Ins) => (Sub, first.original.clone(), second.correction.clone()),
        (Del, Del) => (Del, join(&first.original, &second.original), String::new()),
        (Del, Sub | Noop) => (
            Sub,
            join(&first.original, &second.original),
            second.correction.clone(),
        ),
        (Sub, Ins) => (
            Sub,
            first.original.clone(),
            join(&first.correction, &second.correction),
        ),
        (Sub, Del) => (
            Sub,
            join(&first.original, &second.original),
            first.correction.clone(),
        ),
        (Sub, Sub | Noop) => (
            Sub,
            join(&first.original, &second.original),
            join(&first.correction, &second.correction),
        ),
        (Noop, Ins) => (
            Sub,
            first.original.clone(),
            join(&first.correction, &second.correction),
        ),
        (Noop, Del) => (
            Sub,
            join(&first.original, &second.original),
            first.correction.clone(),
        ),
        (Noop, Sub) => (
            Sub,
            join(&first.original, &second.original),
            join(&first.correction, &second.correction),
        ),
        (Noop, Noop) => (
            Noop,
            join(&first.original, &second.original),
            join(&first.correction, &second.correction),
        ),
    };

    GraphEdit {
        kind,
        start: first.start,
        end: second.end,
        original,
        correction,
        unchanged: first.unchanged + second.unchanged,
    }
}

fn join(first: &str, second: &str) -> String {
    match (first.is_empty(), second.is_empty()) {
        (true, true) => String::new(),
        (true, false) => second.to_string(),
        (false, true) => first.to_string(),
        (false, false) => format!("{first} {second}"),
    }
}

fn add_transitive_arcs(graph: &mut Graph) {
    let vertices = graph.vertices.clone();
    for &middle in &vertices {
        for &from in &vertices {
            let Some(first) = graph.edge(from, middle).cloned() else {
                continue;
            };
            for &to in &vertices {
                let Some(second) = graph.edge(middle, to).cloned() else {
                    continue;
                };
                let distance = first.weight + second.weight;
                let existing = graph
                    .edge(from, to)
                    .map(|edge| edge.weight)
                    .unwrap_or(f64::INFINITY);
                if distance < existing {
                    let edit = merge_edits(&first.edit, &second.edit);
                    if edit.unchanged <= MAX_UNCHANGED_WORDS {
                        graph.add_edge(from, to, edit, distance);
                    }
                }
            }
        }
    }
    graph
        .edges
        .retain(|edge| !(edge.edit.is_noop() && edge.weight > 1.0));
    graph.rebuild_index();
}

fn set_weights(graph: &mut Graph, source: &[String], gold: &[GoldEdit]) {
    let gold = gold_views(source, gold);
    let mut gold_by_span: BTreeMap<Span, Vec<usize>> = BTreeMap::new();
    for (index, edit) in gold.iter().enumerate() {
        gold_by_span
            .entry((edit.start, edit.end))
            .or_default()
            .push(index);
    }

    let mut model_by_span: BTreeMap<Span, Vec<usize>> = BTreeMap::new();
    for (index, edge) in graph.edges.iter().enumerate() {
        model_by_span
            .entry((edge.edit.start, edge.edit.end))
            .or_default()
            .push(index);
    }
    let edge_count = graph.edges.len() as f64;

    for (span, mut model_edges) in model_by_span {
        model_edges.sort_unstable_by_key(|&index| {
            let edge = &graph.edges[index];
            (edge.from, edge.to)
        });
        let gold_edges = gold_by_span.get(&span).cloned().unwrap_or_default();
        if span.0 != span.1 {
            for edge_index in model_edges {
                let matches = gold_edges.iter().any(|&gold_index| {
                    matches_gold(&graph.edges[edge_index].edit, &gold[gold_index])
                });
                if matches {
                    graph.edges[edge_index].weight = -edge_count;
                } else if !graph.edges[edge_index].edit.is_noop() {
                    graph.edges[edge_index].weight += EPSILON;
                }
            }
            continue;
        }

        // Python's insertion matching consumes gold insertions from the outside in so
        // that a sequence of insertions at the same source position cannot reuse one gold
        // edit. The exclusive bounds make the original pointer dance safe in Rust.
        let mut left = 0usize;
        let mut right = model_edges.len();
        let mut current = left;
        let mut gold_left = 0usize;
        let mut gold_right = gold_edges.len();
        while left < right {
            let at_left = current == left;
            let current_index = model_edges[current];
            let current_edit = graph.edges[current_index].edit.clone();
            let matching_gold = if at_left {
                (gold_left..gold_right)
                    .find(|&gold_index| matches_gold(&current_edit, &gold[gold_edges[gold_index]]))
            } else {
                (gold_left..gold_right)
                    .rev()
                    .find(|&gold_index| matches_gold(&current_edit, &gold[gold_edges[gold_index]]))
            };

            if let Some(gold_index) = matching_gold {
                graph.edges[current_index].weight = -edge_count;
                if at_left {
                    gold_left = gold_index + 1;
                    left += 1;
                    while left < right
                        && graph.edges[model_edges[left]].from != graph.edges[current_index].to
                    {
                        if !graph.edges[model_edges[left]].edit.is_noop() {
                            graph.edges[model_edges[left]].weight += EPSILON;
                        }
                        left += 1;
                    }
                    current = left;
                } else {
                    gold_right = gold_index;
                    right -= 1;
                    while right > 0
                        && graph.edges[model_edges[right - 1]].to != graph.edges[current_index].from
                    {
                        if !graph.edges[model_edges[right - 1]].edit.is_noop() {
                            graph.edges[model_edges[right - 1]].weight += EPSILON;
                        }
                        right -= 1;
                    }
                    current = right.saturating_sub(1);
                }
            } else if at_left {
                if !current_edit.is_noop() {
                    graph.edges[current_index].weight += EPSILON;
                }
                left += 1;
                current = right - 1;
            } else {
                if !current_edit.is_noop() {
                    graph.edges[current_index].weight += EPSILON;
                }
                right -= 1;
                current = left;
            }
        }
    }
}

fn best_edit_sequence(graph: &Graph) -> Vec<GraphEdit> {
    let mut distance = HashMap::<Vertex, f64>::new();
    let mut path = HashMap::<Vertex, Vertex>::new();
    for &vertex in &graph.vertices {
        distance.insert(vertex, f64::INFINITY);
    }
    distance.insert((0, 0), 0.0);

    for _ in 0..graph.vertices.len().saturating_sub(1) {
        for edge in &graph.edges {
            let from_distance = distance[&edge.from];
            let candidate = from_distance + edge.weight;
            if candidate < distance[&edge.to] {
                distance.insert(edge.to, candidate);
                path.insert(edge.to, edge.from);
            }
        }
    }

    let mut vertex = *graph
        .vertices
        .iter()
        .max()
        .expect("edit graph is not empty");
    let mut sequence = Vec::new();
    while let Some(&previous) = path.get(&vertex) {
        let edge_index = graph
            .edge_index
            .get(&(previous, vertex))
            .expect("path edge is present");
        let edit = &graph.edges[*edge_index].edit;
        if !edit.is_noop() {
            sequence.push(edit.clone());
        }
        vertex = previous;
    }
    sequence
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_string).collect()
    }

    fn reference(edits: &[(usize, usize, &str)]) -> Reference {
        Reference {
            annotator: 0,
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
    fn scores_a_phrase_longer_than_the_corpus_gold_limit() {
        let score = score(
            &tokens("a b c d e f g h i"),
            &tokens("x y z q r s t h i"),
            &[reference(&[(0, 7, "x y z q r s t")])],
        );

        assert_eq!(
            (score.counts.tp, score.counts.fp, score.counts.fneg),
            (1, 0, 0)
        );
        assert_eq!(score.edits.len(), 1);
        assert_eq!(score.edits[0].start, 0);
        assert_eq!(score.edits[0].end, 7);
    }

    #[test]
    fn scores_multiple_insertions_at_one_source_position() {
        let score = score(
            &tokens("a b"),
            &tokens("x y a b"),
            &[Reference {
                annotator: 0,
                sentence: String::new(),
                edits: vec![
                    GoldEdit {
                        start: 0,
                        end: 0,
                        replacements: vec!["x".to_string()],
                    },
                    GoldEdit {
                        start: 0,
                        end: 0,
                        replacements: vec!["y".to_string()],
                    },
                ],
            }],
        );

        assert_eq!(
            (score.counts.tp, score.counts.fp, score.counts.fneg),
            (2, 0, 0)
        );
        assert_eq!(score.edits.len(), 2);
        assert_eq!(score.edits[0].replacement, tokens("x"));
        assert_eq!(score.edits[1].replacement, tokens("y"));
    }

    #[test]
    fn picks_the_reference_with_the_best_f05() {
        let references = [
            reference(&[(1, 2, "x"), (2, 3, "y")]),
            reference(&[(1, 2, "x")]),
        ];
        let score = score(&tokens("a b c"), &tokens("a x c"), &references);

        assert_eq!(score.reference, 1);
        assert_eq!(
            (score.counts.tp, score.counts.fp, score.counts.fneg),
            (1, 0, 0)
        );
    }

    #[test]
    fn penalizes_unmatched_insertions_so_transitive_phrase_wins() {
        let score = score(
            &tokens("저는 choco-pie를 좋아해요 ."),
            &tokens("저는 초콜릿 파이를 좋아해요 ."),
            &[reference(&[])],
        );

        assert_eq!(score.counts.fp, 1);
        assert_eq!(score.edits.len(), 1);
        assert_eq!(score.edits[0].start, 1);
        assert_eq!(score.edits[0].end, 3);
    }
}
