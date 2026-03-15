from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schemas import (
    Demographics,
    NetworkLayer,
    PastActivity,
    Persona,
    SocialGroup,
    SocialWorld,
    StaticFact,
)

DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "data" / "input"
DEFAULT_INPUT_FILE = DEFAULT_INPUT_DIR / "social_world.json"


def _load_json(path: str | Path) -> dict[str, Any]:
    with resolve_input_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_input_path(path: str | Path | None = None) -> Path:
    if path is None:
        candidate = DEFAULT_INPUT_FILE
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"No input file found. Put social_world.json at {DEFAULT_INPUT_FILE} or pass --input PATH."
        )

    requested = Path(path).expanduser()
    if requested.exists():
        return requested.resolve()

    repo_input_candidate = DEFAULT_INPUT_DIR / requested.name
    if repo_input_candidate.exists():
        return repo_input_candidate.resolve()

    raise FileNotFoundError(
        f"Could not find input JSON at {requested} or {repo_input_candidate}."
    )


def _parse_demographics(raw: dict[str, Any]) -> Demographics:
    return Demographics(
        name=str(raw.get("name", "")),
        age=int(raw.get("age", 0)),
        gender=str(raw.get("gender", "")),
        education=str(raw.get("education", "")),
        occupation=str(raw.get("occupation", "")),
        income_bracket=str(raw.get("income_bracket", "")),
        location_type=str(raw.get("location_type", "")),
        marital_status=str(raw.get("marital_status", "")),
    )


def _parse_static_facts(raw_facts: list[dict[str, Any]]) -> list[StaticFact]:
    return [
        StaticFact(
            id=str(fact.get("id", "")),
            category=str(fact.get("category", "")),
            content=str(fact.get("content", "")),
            evidence_items=[str(item) for item in fact.get("evidence_items", [])],
        )
        for fact in raw_facts
        if isinstance(fact, dict)
    ]


def _parse_past_activities(raw_activities: list[dict[str, Any]]) -> list[PastActivity]:
    return [
        PastActivity(
            id=str(activity.get("id", "")),
            event_type=str(activity.get("event_type", "")),
            content=str(activity.get("content", "")),
            location=str(activity.get("location", "")),
            timestamp=str(activity.get("timestamp", "")),
            involved_entities=[str(entity) for entity in activity.get("involved_entities", [])],
            evidence_items=[str(item) for item in activity.get("evidence_items", [])],
        )
        for activity in raw_activities
        if isinstance(activity, dict)
    ]


def _parse_social_network(raw_network: dict[str, Any]) -> dict[str, NetworkLayer]:
    parsed: dict[str, NetworkLayer] = {}
    for layer_name, layer_data in raw_network.items():
        if not isinstance(layer_data, dict):
            continue
        parsed[layer_name] = NetworkLayer(
            description=str(layer_data.get("description", "")),
            base_layer=layer_data.get("base_layer"),
            scaled_size=layer_data.get("scaled_size"),
            members=[str(member) for member in layer_data.get("members", [])],
        )
    return parsed


def _parse_people(raw_people: list[dict[str, Any]]) -> list[Persona]:
    people: list[Persona] = []
    for raw_person in raw_people:
        hidden_context = raw_person.get("hidden_context_ground_truth", {})
        people.append(
            Persona(
                persona_id=str(raw_person.get("persona_id", "")),
                demographics=_parse_demographics(raw_person.get("demographics", {})),
                psychological_traits_ocean=dict(raw_person.get("psychological_traits_ocean", {})),
                static_facts=_parse_static_facts(hidden_context.get("static_facts", [])),
                past_activities=_parse_past_activities(hidden_context.get("past_activities", [])),
                app_log_blueprints=list(raw_person.get("app_log_blueprints", [])),
                social_network=_parse_social_network(raw_person.get("social_network", {})),
            )
        )
    return people


def _parse_groups(raw_groups: list[dict[str, Any]]) -> list[SocialGroup]:
    groups: list[SocialGroup] = []
    for raw_group in raw_groups:
        groups.append(
            SocialGroup(
                group_id=str(raw_group.get("group_id", "")),
                group_name=str(raw_group.get("group_name", "")),
                group_type=str(raw_group.get("type", "")),
                members=[str(member) for member in raw_group.get("members", [])],
                hidden_group_context=dict(raw_group.get("hidden_group_context", {})),
            )
        )
    return groups


def load_social_world(path: str | Path) -> SocialWorld:
    raw_data = _load_json(path)
    return SocialWorld(
        metadata=dict(raw_data.get("metadata", {})),
        people=_parse_people(raw_data.get("people", [])),
        social_groups=_parse_groups(raw_data.get("social_groups", [])),
    )


def validate_social_world_structure(world: SocialWorld) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    people_by_id = world.people_by_id

    if not world.people:
        findings.append(
            {
                "severity": "error",
                "message": "The social world contains no personas.",
                "fields": ["people"],
            }
        )

    duplicate_persona_ids = len(world.people) != len(people_by_id)
    if duplicate_persona_ids:
        findings.append(
            {
                "severity": "error",
                "message": "Duplicate persona_id values were found.",
                "fields": ["people[].persona_id"],
            }
        )

    for persona in world.people:
        if not persona.persona_id:
            findings.append(
                {
                    "severity": "error",
                    "message": "A persona is missing persona_id.",
                    "fields": ["people[].persona_id"],
                }
            )
        if not persona.demographics.name:
            findings.append(
                {
                    "severity": "warning",
                    "message": f"{persona.persona_id} is missing demographics.name.",
                    "fields": [f"people[persona_id={persona.persona_id}].demographics.name"],
                }
            )

    for group in world.social_groups:
        missing_members = [member for member in group.members if member not in people_by_id]
        if missing_members:
            findings.append(
                {
                    "severity": "error",
                    "message": f"{group.group_id} references persona IDs that are not present in people.",
                    "fields": [f"social_groups[group_id={group.group_id}].members"],
                    "missing_members": missing_members,
                }
            )

    return findings


def summarize_schema(world: SocialWorld) -> dict[str, Any]:
    return {
        "top_level_keys": ["metadata", "people", "social_groups"],
        "people_count": len(world.people),
        "social_groups_count": len(world.social_groups),
        "persona_fields": [
            "persona_id",
            "demographics",
            "psychological_traits_ocean",
            "hidden_context_ground_truth.static_facts",
            "hidden_context_ground_truth.past_activities",
            "app_log_blueprints",
            "social_network",
        ],
        "demographics_fields": [
            "name",
            "age",
            "gender",
            "education",
            "occupation",
            "income_bracket",
            "location_type",
            "marital_status",
        ],
        "social_network_layers": sorted(
            {
                layer_name
                for persona in world.people
                for layer_name in persona.social_network.keys()
            }
        ),
        "static_fact_fields": ["id", "category", "content", "evidence_items"],
        "past_activity_fields": [
            "id",
            "event_type",
            "content",
            "location",
            "timestamp",
            "involved_entities",
            "evidence_items",
        ],
        "notes": [
            "The shared multi-persona dataset is social_world.json.",
            "persona.json in Downloads is a commented reference spec, not valid JSON input.",
            "No explicit spouse_id, household_id, parent_id, or child_id fields are present.",
        ],
    }
