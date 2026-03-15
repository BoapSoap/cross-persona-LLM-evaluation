from __future__ import annotations

import argparse
from pathlib import Path

from link_extractor import create_evaluation_plan
from llm_eval import (
    PAIR_EVAL_RESPONSE_SCHEMA,
    build_sample_request_payload,
    evaluate_pairs_with_openai,
)
from loader import (
    DEFAULT_INPUT_FILE,
    load_social_world,
    resolve_input_path,
    summarize_schema,
    validate_social_world_structure,
)
from report_writer import build_dry_run_report, build_llm_report, write_report
from schemas import VerificationConfig


def parse_args() -> VerificationConfig:
    parser = argparse.ArgumentParser(
        description="LLM-first cross-persona relationship verifier for shared social-world JSON files."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_FILE.relative_to(Path(__file__).resolve().parent)),
        help="Path to social_world.json. Defaults to data/input/social_world.json.",
    )
    parser.add_argument(
        "--output",
        default="output/social_world_verification.json",
        help="Path to write the structured JSON report.",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4.1-mini",
        help="OpenAI model to use for pair evaluation.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=200,
        help="Maximum number of selected persona pairs to evaluate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the API; only show which pairs would be evaluated and emit a sample request payload.",
    )
    parser.add_argument(
        "--max-shared-group-size",
        type=int,
        default=5,
        help="Only use shared-group-only pairs from groups up to this size unless the pair shares multiple groups.",
    )
    parser.add_argument(
        "--exclude-group-only-pairs",
        action="store_true",
        help="Exclude pairs that are selected only because of shared-group evidence.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1200,
        help="Max output tokens for each Responses API call.",
    )
    parser.add_argument(
        "--max-retries-per-pair",
        type=int,
        default=2,
        help="Retry count when a model response is missing or schema-invalid.",
    )
    args = parser.parse_args()
    return VerificationConfig(
        input_path=args.input,
        output_path=args.output,
        llm_model=args.llm_model,
        max_pairs=args.max_pairs,
        dry_run=args.dry_run,
        max_shared_group_size=args.max_shared_group_size,
        include_group_only_pairs=not args.exclude_group_only_pairs,
        max_output_tokens=args.max_output_tokens,
        max_retries_per_pair=args.max_retries_per_pair,
    )


def _planned_pair_entry(pair: object) -> dict[str, object]:
    return {
        "pair_id": pair.pair_id,
        "personas": list(pair.personas),
        "priority_score": pair.priority_score,
        "selection_reasons": pair.selection_reasons,
        "source_fields": pair.source_fields,
    }


def main() -> None:
    config = parse_args()
    resolved_input_path = resolve_input_path(config.input_path)
    world = load_social_world(resolved_input_path)
    schema_summary = summarize_schema(world)
    structure_validation = validate_social_world_structure(world)

    plan = create_evaluation_plan(
        world,
        max_pairs=config.max_pairs,
        max_shared_group_size=config.max_shared_group_size,
        include_group_only_pairs=config.include_group_only_pairs,
    )
    sample_request_payload = (
        build_sample_request_payload(world, plan.selected_pairs[0])
        if plan.selected_pairs
        else None
    )

    if config.dry_run:
        report = build_dry_run_report(
            input_path=str(resolved_input_path),
            schema_summary=schema_summary,
            structure_validation=structure_validation,
            selection_summary=plan.selection_summary,
            selected_pairs=[_planned_pair_entry(pair) for pair in plan.selected_pairs],
            sample_request_payload=sample_request_payload,
        )
        write_report(report, config.output_path)
        print(
            "Dry run:",
            f"candidate_pairs={plan.selection_summary['candidate_pair_count']}",
            f"selected_pairs={plan.selection_summary['selected_pair_count']}",
        )
        print(f"Structured output schema keys: {list(PAIR_EVAL_RESPONSE_SCHEMA['properties'].keys())}")
        print(f"Report written to {Path(config.output_path).resolve()}")
        return

    llm_results = evaluate_pairs_with_openai(
        world,
        plan.selected_pairs,
        model=config.llm_model,
        max_output_tokens=config.max_output_tokens,
        max_retries_per_pair=config.max_retries_per_pair,
    )
    report = build_llm_report(
        input_path=str(resolved_input_path),
        model=config.llm_model,
        schema_summary=schema_summary,
        structure_validation=structure_validation,
        selection_summary=plan.selection_summary,
        evaluations=llm_results["evaluations"],
    )
    write_report(report, config.output_path)

    aggregate_counts = report["aggregate_counts"]
    print(
        "Evaluation summary:",
        f"evaluated={llm_results['evaluated_pair_count']}",
        f"Pass={aggregate_counts.get('Pass', 0)}",
        f"Warning={aggregate_counts.get('Warning', 0)}",
        f"Fail={aggregate_counts.get('Fail', 0)}",
    )
    print(f"Report written to {Path(config.output_path).resolve()}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
