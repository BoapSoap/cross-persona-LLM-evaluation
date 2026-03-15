from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from link_extractor import build_request_payload
from schemas import CandidatePair, SocialWorld


PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "pair_eval_prompt.txt"
DEFAULT_ENV_PATH = Path(__file__).resolve().parent / ".env"

PAIR_EVAL_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "pair_id",
        "overall_label",
        "dimension_scores",
        "issues",
        "summary",
    ],
    "properties": {
        "pair_id": {"type": "string"},
        "overall_label": {
            "type": "string",
            "enum": ["Pass", "Warning", "Fail"],
        },
        "dimension_scores": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "reciprocal_relationship_evidence",
                "relationship_strength_alignment",
                "age_and_life_stage_plausibility",
                "location_and_interaction_plausibility",
                "shared_activity_consistency",
                "conflict_consistency",
                "household_partner_family_plausibility",
                "overall_social_world_coherence",
            ],
            "properties": {
                "reciprocal_relationship_evidence": {
                    "type": "string",
                    "enum": ["Pass", "Warning", "Fail"],
                },
                "relationship_strength_alignment": {
                    "type": "string",
                    "enum": ["Pass", "Warning", "Fail"],
                },
                "age_and_life_stage_plausibility": {
                    "type": "string",
                    "enum": ["Pass", "Warning", "Fail"],
                },
                "location_and_interaction_plausibility": {
                    "type": "string",
                    "enum": ["Pass", "Warning", "Fail"],
                },
                "shared_activity_consistency": {
                    "type": "string",
                    "enum": ["Pass", "Warning", "Fail"],
                },
                "conflict_consistency": {
                    "type": "string",
                    "enum": ["Pass", "Warning", "Fail"],
                },
                "household_partner_family_plausibility": {
                    "type": "string",
                    "enum": ["Pass", "Warning", "Fail"],
                },
                "overall_social_world_coherence": {
                    "type": "string",
                    "enum": ["Pass", "Warning", "Fail"],
                },
            },
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "category",
                    "severity",
                    "description",
                    "evidence_fields",
                ],
                "properties": {
                    "category": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["Low", "Medium", "High"],
                    },
                    "description": {"type": "string"},
                    "evidence_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "summary": {"type": "string"},
    },
}


class DimensionScoresModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reciprocal_relationship_evidence: Literal["Pass", "Warning", "Fail"]
    relationship_strength_alignment: Literal["Pass", "Warning", "Fail"]
    age_and_life_stage_plausibility: Literal["Pass", "Warning", "Fail"]
    location_and_interaction_plausibility: Literal["Pass", "Warning", "Fail"]
    shared_activity_consistency: Literal["Pass", "Warning", "Fail"]
    conflict_consistency: Literal["Pass", "Warning", "Fail"]
    household_partner_family_plausibility: Literal["Pass", "Warning", "Fail"]
    overall_social_world_coherence: Literal["Pass", "Warning", "Fail"]


class IssueModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    severity: Literal["Low", "Medium", "High"]
    description: str
    evidence_fields: list[str]


class PairEvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair_id: str
    overall_label: Literal["Pass", "Warning", "Fail"]
    dimension_scores: DimensionScoresModel
    issues: list[IssueModel]
    summary: str


def load_prompt_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def load_env_from_file(env_path: Path = DEFAULT_ENV_PATH) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def build_prompt(payload: dict[str, Any], prompt_template: str | None = None) -> str:
    template = prompt_template or load_prompt_template()
    return (
        template.replace(
            "{{PERSONA_A_JSON}}",
            json.dumps(payload["persona_a"], indent=2, ensure_ascii=True),
        ).replace(
            "{{PERSONA_B_JSON}}",
            json.dumps(payload["persona_b"], indent=2, ensure_ascii=True),
        )
    )


def validate_structured_output(
    raw_text: str,
    *,
    expected_pair_id: str,
) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON for {expected_pair_id}: {exc}") from exc

    try:
        validated = PairEvaluationModel.model_validate(parsed)
    except ValidationError as exc:
        raise ValueError(
            f"Model returned schema-invalid JSON for {expected_pair_id}: {exc}"
        ) from exc

    if validated.pair_id != expected_pair_id:
        raise ValueError(
            f"Model returned pair_id={validated.pair_id}, expected {expected_pair_id}."
        )

    return validated.model_dump()


def build_sample_request_payload(
    world: SocialWorld,
    candidate_pair: CandidatePair,
) -> dict[str, Any]:
    return build_request_payload(world, candidate_pair)


def evaluate_pairs_with_openai(
    world: SocialWorld,
    candidate_pairs: list[CandidatePair],
    *,
    model: str,
    max_output_tokens: int,
    max_retries_per_pair: int,
) -> dict[str, Any]:
    load_env_from_file()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Use dry-run mode or export the API key to run the default LLM-first pipeline."
        )

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    prompt_template = load_prompt_template()
    evaluations: list[dict[str, Any]] = []
    request_payloads: list[dict[str, Any]] = []

    for candidate_pair in candidate_pairs:
        payload = build_request_payload(world, candidate_pair)
        request_payloads.append(payload)
        prompt = build_prompt(payload, prompt_template)
        last_error: Exception | None = None

        for _attempt in range(max_retries_per_pair + 1):
            try:
                response = client.responses.create(
                    model=model,
                    input=prompt,
                    max_output_tokens=max_output_tokens,
                    temperature=0,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "pair_relationship_evaluation",
                            "schema": PAIR_EVAL_RESPONSE_SCHEMA,
                            "strict": True,
                        }
                    },
                )
                raw_text = getattr(response, "output_text", "")
                if not raw_text:
                    raise ValueError(
                        f"Empty structured output returned for {candidate_pair.pair_id}."
                    )
                validated = validate_structured_output(
                    raw_text,
                    expected_pair_id=candidate_pair.pair_id,
                )
                evaluations.append(
                    {
                        "pair_id": validated["pair_id"],
                        "personas": list(candidate_pair.personas),
                        "overall_label": validated["overall_label"],
                        "dimension_scores": validated["dimension_scores"],
                        "issues": validated["issues"],
                        "summary": validated["summary"],
                    }
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        else:
            raise RuntimeError(
                f"Failed to obtain a valid structured response for {candidate_pair.pair_id}: {last_error}"
            ) from last_error

    return {
        "model": model,
        "evaluated_pair_count": len(evaluations),
        "evaluations": evaluations,
        "request_payloads": request_payloads,
    }
