from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any


def canonical_pair(persona_a: str, persona_b: str) -> tuple[str, str]:
    return tuple(sorted((persona_a, persona_b)))


def pair_id_for(persona_a: str, persona_b: str) -> str:
    left_id, right_id = canonical_pair(persona_a, persona_b)
    return f"{left_id}__{right_id}"


def extract_city_region(location: str | None) -> str | None:
    if not location:
        return None

    match = re.search(r"\(([^)]+)\)", location)
    if match:
        return match.group(1).strip()

    parts = [part.strip() for part in location.split(",") if part.strip()]
    if len(parts) >= 2:
        return ", ".join(parts[-2:])
    if parts:
        return parts[-1]
    return None


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def field_path_for_persona(persona_id: str, suffix: str) -> str:
    return f"people[persona_id={persona_id}].{suffix}"


def field_path_for_group(group_id: str) -> str:
    return f"social_groups[group_id={group_id}]"


@dataclass(frozen=True)
class Demographics:
    name: str
    age: int
    gender: str
    education: str
    occupation: str
    income_bracket: str
    location_type: str
    marital_status: str

    @property
    def city_region(self) -> str | None:
        return extract_city_region(self.location_type)


@dataclass(frozen=True)
class StaticFact:
    id: str
    category: str
    content: str
    evidence_items: list[str]


@dataclass(frozen=True)
class PastActivity:
    id: str
    event_type: str
    content: str
    location: str
    timestamp: str
    involved_entities: list[str]
    evidence_items: list[str]

    @property
    def city_region(self) -> str | None:
        return extract_city_region(self.location)

    @property
    def parsed_timestamp(self) -> datetime | None:
        return parse_timestamp(self.timestamp)


@dataclass(frozen=True)
class NetworkLayer:
    description: str
    base_layer: int | None
    scaled_size: int | None
    members: list[str]


@dataclass(frozen=True)
class Persona:
    persona_id: str
    demographics: Demographics
    psychological_traits_ocean: dict[str, float]
    static_facts: list[StaticFact]
    past_activities: list[PastActivity]
    app_log_blueprints: list[dict[str, Any]]
    social_network: dict[str, NetworkLayer]

    def relationship_layer_for(self, other_persona_id: str) -> str | None:
        for layer_name, layer in self.social_network.items():
            if other_persona_id in layer.members:
                return layer_name
        return None


@dataclass(frozen=True)
class SocialGroup:
    group_id: str
    group_name: str
    group_type: str
    members: list[str]
    hidden_group_context: dict[str, Any]


@dataclass(frozen=True)
class SocialWorld:
    metadata: dict[str, Any]
    people: list[Persona]
    social_groups: list[SocialGroup]

    @property
    def people_by_id(self) -> dict[str, Persona]:
        return {persona.persona_id: persona for persona in self.people}


@dataclass
class VerificationConfig:
    input_path: str
    output_path: str
    llm_model: str = "gpt-4.1-mini"
    max_pairs: int = 200
    dry_run: bool = False
    max_shared_group_size: int = 5
    include_group_only_pairs: bool = True
    max_output_tokens: int = 1200
    max_retries_per_pair: int = 2


@dataclass
class CandidatePair:
    pair_id: str
    personas: tuple[str, str]
    selection_reasons: list[str] = field(default_factory=list)
    priority_score: int = 0
    source_fields: list[str] = field(default_factory=list)
    network_layers: dict[str, str | None] = field(default_factory=dict)
    shared_groups: list[dict[str, Any]] = field(default_factory=list)
    directed_activities: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    directed_conflicts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


@dataclass
class EvaluationPlan:
    candidate_pairs: list[CandidatePair]
    selected_pairs: list[CandidatePair]
    selection_summary: dict[str, Any]
