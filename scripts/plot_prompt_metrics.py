# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "pandas"]
# ///
"""Plot how prompt length, prompt language and message layout move the F0.5 of each model.

Reads the ``prompt-metrics.csv`` written by ``collect_prompt_metrics.py``, so collect first:

    uv run --no-project python ./scripts/collect_prompt_metrics.py --results-dir multiprompt-results
    uv run ./scripts/plot_prompt_metrics.py            # writes PNGs to multiprompt-results/plots
    uv run ./scripts/plot_prompt_metrics.py --show     # also opens interactive windows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

GROUP_COLORS = {"hosted": "#2a78d6", "local": "#eb6834"}
PROMPT_COLORS = {"simple": "#2a78d6", "extended": "#eb6834", "long": "#1baf7a", "korean": "#eda100"}
TYPE_MARKERS = {"user": "o", "system+user": "s"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update(
    {
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "font.size": 9,
        "legend.frameon": False,
    }
)


def load_runs(metrics_file: Path, prompt_file: Path) -> pd.DataFrame:
    """One row per model and prompt variant (latest run wins), with group and prompt length."""
    runs = pd.read_csv(metrics_file)
    runs = runs[runs["prompt_name"] != "custom"]
    runs = runs.sort_values("started").drop_duplicates(["model", "prompt_name", "prompt_type"], keep="last")
    runs["group"] = runs["provider"].map(lambda p: "local" if p == "lmstudio" else "hosted")
    runs["variant"] = runs["prompt_name"] + "/" + runs["prompt_type"]

    # Instruction length: every character the model sees besides the sentence itself.
    lengths = {}
    for prompt in json.loads(prompt_file.read_text(encoding="utf-8")):
        text = (prompt.get("system") or "") + prompt["user"].replace("{sentence}", "")
        lengths[prompt["name"]] = len(text.strip())
    runs["prompt_chars"] = runs["prompt_name"].map(lengths)
    return runs.reset_index(drop=True)


def model_order(runs: pd.DataFrame) -> list[str]:
    """Hosted models first, each group sorted by median F0.5, best on top."""
    medians = runs.groupby(["group", "model"])["f05"].median().reset_index()
    medians["group_rank"] = medians["group"].map({"hosted": 0, "local": 1})
    return medians.sort_values(["group_rank", "f05"], ascending=[True, False])["model"].tolist()


def plot_length(runs: pd.DataFrame) -> plt.Figure:
    """F0.5 against instruction length, one line per model; Korean is a separate marker."""
    # Ordinal x axis, shortest prompt first: the lengths are too uneven (tens vs hundreds of
    # characters) for a linear axis to keep the short prompts apart.
    lengths = runs.groupby("prompt_name")["prompt_chars"].first().sort_values()
    position = {name: i for i, name in enumerate(lengths.index)}
    tick_labels = [f"{name}\n{chars} chars" for name, chars in lengths.items()]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, prompt_type in zip(axes, TYPE_MARKERS):
        subset = runs[runs["prompt_type"] == prompt_type]
        labels = []
        for model, model_runs in subset.groupby("model"):
            color = GROUP_COLORS[model_runs["group"].iloc[0]]
            english = model_runs[model_runs["prompt_name"] != "korean"].sort_values("prompt_chars")
            korean = model_runs[model_runs["prompt_name"] == "korean"]
            ax.plot(english["prompt_name"].map(position), english["f05"], color=color, lw=2, marker="o", ms=6)
            ax.scatter(korean["prompt_name"].map(position), korean["f05"], color="white", edgecolor=color,
                       lw=2, marker="D", s=50, zorder=3)
            if not english.empty:
                labels.append([english["f05"].iloc[-1], model.split("/")[-1]])
        # Push apart labels closer than a minimum gap so neighbouring models stay readable.
        labels.sort()
        gap = 0.03
        for below, above in zip(labels, labels[1:]):
            above[0] = max(above[0], below[0] + gap)
        for y, name in labels:
            ax.annotate(name, (len(position) - 1, y), xytext=(10, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK, annotation_clip=False)
        ax.set_xticks(range(len(position)), tick_labels)
        ax.set_title(f"Prompt sent as {prompt_type}", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("prompt, ordered by instruction length (excluding the sentence)")
        ax.set_xlim(-0.4, len(position) - 1 + 1.6)
    axes[0].set_ylabel("F0.5")
    handles = [plt.Line2D([], [], color=c, lw=2, marker="o", label=g) for g, c in GROUP_COLORS.items()]
    handles.append(plt.Line2D([], [], color=MUTED, lw=0, marker="D", mfc="white", mew=2, label="Korean prompt"))
    fig.legend(handles=handles, loc="upper right", ncol=3)
    fig.suptitle("Prompt length vs quality (English prompts connected, shortest to longest)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_range(runs: pd.DataFrame) -> plt.Figure:
    """Per model: the F0.5 range across all prompt variants, each variant a dot."""
    order = model_order(runs)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(order) + 1.8))
    for y, model in enumerate(order):
        model_runs = runs[runs["model"] == model]
        low, high = model_runs["f05"].min(), model_runs["f05"].max()
        group = model_runs["group"].iloc[0]
        ax.hlines(y, low, high, color=GROUP_COLORS[group], lw=6, alpha=0.25)
        for _, run in model_runs.iterrows():
            ax.scatter(run["f05"], y, color=PROMPT_COLORS.get(run["prompt_name"], MUTED),
                       marker=TYPE_MARKERS[run["prompt_type"]], s=55, edgecolor="#fcfcfb", lw=1.5, zorder=3)
        ax.annotate(f"range {high - low:.3f}", (high, y), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(order)), [f"{m}  ({runs.loc[runs['model'] == m, 'group'].iloc[0]})" for m in order])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("F0.5")
    ax.margins(x=0.12)
    handles = [plt.Line2D([], [], color=c, lw=0, marker="o", ms=7, label=n) for n, c in PROMPT_COLORS.items()]
    handles += [plt.Line2D([], [], color=MUTED, lw=0, marker=m, ms=7, label=t) for t, m in TYPE_MARKERS.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14 - 0.3 / len(order)), ncol=6)
    ax.set_title("What range each model covers across prompts", loc="left", fontsize=12)
    fig.tight_layout()
    return fig


def plot_language(runs: pd.DataFrame) -> plt.Figure:
    """Korean prompt F0.5 minus the mean and the best English prompt, per model and layout."""
    order = model_order(runs)
    rows = []
    for (model, prompt_type), model_runs in runs.groupby(["model", "prompt_type"]):
        english = model_runs.loc[model_runs["prompt_name"] != "korean", "f05"]
        korean = model_runs.loc[model_runs["prompt_name"] == "korean", "f05"]
        if english.empty or korean.empty:
            continue
        rows.append({"model": model, "prompt_type": prompt_type, "group": model_runs["group"].iloc[0],
                     "vs_mean": korean.iloc[0] - english.mean(), "vs_best": korean.iloc[0] - english.max()})
    deltas = pd.DataFrame(rows)
    order = [m for m in order if m in set(deltas["model"])]

    fig, axes = plt.subplots(1, 2, figsize=(11, 0.5 * len(order) + 2), sharey=True)
    for ax, prompt_type in zip(axes, TYPE_MARKERS):
        subset = deltas[deltas["prompt_type"] == prompt_type].set_index("model").reindex(order)
        ys = range(len(order))
        colors = [GROUP_COLORS.get(g, MUTED) for g in subset["group"]]
        ax.barh([y - 0.18 for y in ys], subset["vs_mean"], height=0.34, color=colors, label="vs mean English")
        ax.barh([y + 0.18 for y in ys], subset["vs_best"], height=0.34, color=colors, alpha=0.4, label="vs best English")
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(f"Prompt sent as {prompt_type}", loc="left", fontsize=10)
        ax.set_xlabel("F0.5 difference (Korean prompt − English prompts)")
        ax.grid(axis="y", visible=False)
        limit = deltas[["vs_mean", "vs_best"]].abs().max().max() * 1.15
        ax.set_xlim(-limit, limit)
    axes[0].set_yticks(range(len(order)), order)
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=MUTED, label="vs mean of English prompts"),
               plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.4, label="vs best English prompt")]
    handles += [plt.Rectangle((0, 0), 1, 1, color=c, label=g) for g, c in GROUP_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4)
    fig.suptitle("Does a Korean prompt help? (right of zero = Korean better)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    return fig


def plot_heatmap(runs: pd.DataFrame) -> plt.Figure:
    """Model x variant: colour is the gap to that model's best prompt, text is the absolute F0.5."""
    order = model_order(runs)
    variants = sorted(runs["variant"].unique(), key=lambda v: (v.split("/")[1], v.split("/")[0]))
    matrix = runs.pivot(index="model", columns="variant", values="f05").reindex(index=order, columns=variants)
    gap = matrix.sub(matrix.max(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(1.1 * len(variants) + 3.5, 0.55 * len(order) + 2.2))
    image = ax.imshow(gap, cmap="Blues", vmin=gap.min().min(), vmax=0, aspect="auto")
    for i, model in enumerate(order):
        for j, variant in enumerate(variants):
            value = matrix.loc[model, variant]
            if pd.notna(value):
                dark = gap.loc[model, variant] > gap.min().min() / 2
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8, color="white" if dark else INK)
    # Mean rank of each variant across models: a prompt that is "high quality" everywhere ranks low.
    ranks = matrix.rank(axis=1, ascending=False).mean()
    ax.set_xticks(range(len(variants)), [f"{v}\nmean rank {ranks[v]:.1f}" for v in variants], rotation=35, ha="right")
    ax.set_yticks(range(len(order)), order)
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="F0.5 gap to the model's best prompt", shrink=0.8)
    ax.set_title("Prompt quality per model (dark = model's best prompt; numbers are F0.5)", loc="left", fontsize=12)
    fig.tight_layout()
    return fig


def plot_sensitivity(runs: pd.DataFrame) -> plt.Figure:
    """Local vs hosted: absolute and relative spread of F0.5 across prompts, against the best F0.5."""
    stats = runs.groupby(["model", "group"])["f05"].agg(["min", "max", "median"]).reset_index()
    stats["spread"] = stats["max"] - stats["min"]
    stats["relative"] = stats["spread"] / stats["max"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, column, label in [(axes[0], "spread", "absolute spread (max − min F0.5)"),
                              (axes[1], "relative", "relative spread ((max − min) / max)")]:
        for group, color in GROUP_COLORS.items():
            subset = stats[stats["group"] == group]
            ax.scatter(subset["max"], subset[column], color=color, s=60, edgecolor="#fcfcfb", lw=1.5, label=group, zorder=3)
            for _, row in subset.iterrows():
                ax.annotate(row["model"].split("/")[-1], (row["max"], row[column]), xytext=(6, 4),
                            textcoords="offset points", fontsize=8, color=INK)
        ax.set_xlabel("best F0.5 over all prompts")
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.margins(x=0.2)
    axes[0].legend(loc="upper right")
    fig.suptitle("Prompt sensitivity: local vs hosted models", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics", type=Path, default=REPOSITORY_ROOT / "multiprompt-results" / "prompt-metrics.csv")
    parser.add_argument("--prompts", type=Path, default=REPOSITORY_ROOT / "scripts" / "prompts.json")
    parser.add_argument("--output-dir", type=Path, help="defaults to a plots/ directory next to --metrics")
    parser.add_argument("--show", action="store_true", help="open the figures in interactive windows")
    args = parser.parse_args()

    runs = load_runs(args.metrics, args.prompts)
    output_dir = args.output_dir or args.metrics.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "1-prompt-length": plot_length(runs),
        "2-prompt-language": plot_language(runs),
        "3-prompt-range": plot_range(runs),
        "4-prompt-quality-heatmap": plot_heatmap(runs),
        "5-local-vs-hosted-sensitivity": plot_sensitivity(runs),
    }
    for name, figure in figures.items():
        path = output_dir / f"{name}.png"
        figure.savefig(path, dpi=150)
        print(f"wrote {path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
