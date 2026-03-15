from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _aggregate_counts(evaluations: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"Pass": 0, "Warning": 0, "Fail": 0}
    for evaluation in evaluations:
        label = evaluation["overall_label"]
        counts[label] = counts.get(label, 0) + 1
    return counts


def build_dry_run_report(
    *,
    input_path: str,
    schema_summary: dict[str, Any],
    structure_validation: list[dict[str, Any]],
    selection_summary: dict[str, Any],
    selected_pairs: list[dict[str, Any]],
    sample_request_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "input_path": input_path,
        "mode": "dry_run",
        "schema_summary": schema_summary,
        "structure_validation": structure_validation,
        "selection_summary": selection_summary,
        "planned_evaluations": selected_pairs,
        "sample_request_payload": sample_request_payload,
    }


def build_llm_report(
    *,
    input_path: str,
    model: str,
    schema_summary: dict[str, Any],
    structure_validation: list[dict[str, Any]],
    selection_summary: dict[str, Any],
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "input_path": input_path,
        "mode": "llm_evaluation",
        "model": model,
        "schema_summary": schema_summary,
        "structure_validation": structure_validation,
        "selection_summary": selection_summary,
        "evaluations": evaluations,
        "aggregate_counts": _aggregate_counts(evaluations),
    }


def write_report(report: dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
