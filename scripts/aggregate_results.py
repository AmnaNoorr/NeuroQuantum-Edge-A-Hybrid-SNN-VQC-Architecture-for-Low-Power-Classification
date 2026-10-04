"""Aggregate per-seed experiment JSON files into a Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METRICS = (
    "accuracy",
    "macro_f1",
    "auroc",
    "parameter_count",
    "inference_time_ms_per_sample",
)


def aggregate(results_dir: Path, output_path: Path) -> None:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    rows: list[dict[str, Any]] = []
    for experiment_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        values_by_metric: dict[str, list[float]] = {metric: [] for metric in METRICS}
        for result_path in sorted(experiment_dir.glob("seed_*.json")):
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            for metric in METRICS:
                value = payload.get("test", {}).get(metric)
                if value is not None:
                    values_by_metric[metric].append(float(value))
        if not any(values_by_metric.values()):
            continue
        row: dict[str, Any] = {"experiment": experiment_dir.name}
        for metric, values in values_by_metric.items():
            row[metric] = _format_mean_std(values)
        rows.append(row)

    headers = ["experiment", *METRICS]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend(
        "| " + " | ".join(str(row.get(header, "n/a")) for header in headers) + " |"
        for row in rows
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved={output_path}")


def _format_mean_std(values: list[float]) -> str:
    if not values:
        return "n/a"
    deviation = stdev(values) if len(values) > 1 else 0.0
    return f"{mean(values):.4f} ± {deviation:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment seed results")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/summary.md"))
    args = parser.parse_args()
    aggregate(args.results_dir, args.output)


if __name__ == "__main__":
    main()