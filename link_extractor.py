from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict
from typing import Any

from schemas import (
    CandidatePair,
    EvaluationPlan,
    SocialWorld,
    canonical_pair,
    field_path_for_group,
    field_path_for_persona,
    pair_id_for,
)


USER_ID_PATTERN = re.compile(r"(USER-\d+)")
SELECTION_REASON_WEIGHTS = {
    "mutual_intimate_circle": 12,
    "mutual_close_friends": 10,
    "one_sided_intimate_circle": 8,
    "one_sided_close_friends": 6,
    "direct_activity_reference": 8,
    "conflict_fact_reference": 9,
    "shared_small_group": 3,
    "shared_multiple_groups": 5,
}


def _activity_summary(persona_id: str, activity: Any, index: int) -> dict[str, Any]:
    return {
        "id": activity.id,
        "event_type": activity.event_type,
        "content": activity.content,
        "location": activity.location,
        "timestamp": activity.timestamp,
        "involved_entities": activity.involved_entities,
        "evidence_items": activity.evidence_items,
        "source_field": field_path_for_persona(
            persona_id, f"hidden_context_ground_truth.past_activities[{index}]"
        ),
    }


def _conflict_summary(persona_id: str, fact: Any, index: int, target_id: str) -> dict[str, Any]:
    return {
        "id": fact.id,
        "category": fact.category,
        "content": fact.content,
        "target_id": target_id,
        "evidence_items": fact.evidence_items,
        "source_field": field_path_for_persona(
            persona_id, f"hidden_context_ground_truth.static_facts[{index}]"
        ),
    }


def _build_pair_payload(world: SocialWorld, pair: CandidatePair) -> dict[str, Any]:
    people_by_id = world.people_by_id
    left_id, right_id = pair.personas
    left_persona = people_by_id[left_id]
    right_persona = people_by_id[right_id]

    def serialize_persona(persona_id: str, other_id: str) -> dict[str, Any]:
        persona = people_by_id[persona_id]
        activities_with_other = pair.directed_activities.get(persona_id, [])
        conflicts_with_other = pair.directed_conflicts.get(persona_id, [])
        recent_activities = sorted(
            [
                {
                    "id": activity.id,
                    "event_type": activity.event_type,
                    "content": activity.content,
                    "location": activity.location,
                    "timestamp": activity.timestamp,
                    "involved_entities": activity.involved_entities,
                    "source_field": field_path_for_persona(
                        persona_id,
                        f"hidden_context_ground_truth.past_activities[{activity_index}]",
                    ),
                }
                for activity_index, activity in enumerate(persona.past_activities)
            ],
            key=lambda item: item["timestamp"],
            reverse=True,
        )[:3]

        intimate_layer = persona.social_network.get("intimate_circle")
        close_layer = persona.social_network.get("close_friends")
        intimate_members = intimate_layer.members if intimate_layer else []
        close_members = close_layer.members if close_layer else []

        return {
            "pair_id": pair.pair_id,
            "persona_id": persona.persona_id,
            "counterparty_persona_id": other_id,
            "demographics": {
                "value": {
                    "name": persona.demographics.name,
                    "age": persona.demographics.age,
                    "gender": persona.demographics.gender,
                    "education": persona.demographics.education,
                    "occupation": persona.demographics.occupation,
                    "income_bracket": persona.demographics.income_bracket,
                    "location_type": persona.demographics.location_type,
                    "marital_status": persona.demographics.marital_status,
                },
                "source_field": field_path_for_persona(persona_id, "demographics"),
            },
            "relationship_evidence_to_counterparty": {
                "social_network_layer": {
                    "value": pair.network_layers.get(persona_id),
                    "source_field": field_path_for_persona(persona_id, "social_network"),
                },
                "counterparty_in_intimate_circle": {
                    "value": other_id in intimate_members,
                    "source_field": field_path_for_persona(
                        persona_id, "social_network.intimate_circle.members"
                    ),
                },
                "counterparty_in_close_friends": {
                    "value": other_id in close_members,
                    "source_field": field_path_for_persona(
                        persona_id, "social_network.close_friends.members"
                    ),
                },
                "static_facts_referencing_counterparty": conflicts_with_other,
                "past_activities_involving_counterparty": activities_with_other,
                "shared_groups_with_counterparty": pair.shared_groups,
            },
            "context_snapshot": {
                "recent_activities": recent_activities,
                "intimate_circle_member_ids": {
                    "value": intimate_members,
                    "source_field": field_path_for_persona(
                        persona_id, "social_network.intimate_circle.members"
                    ),
                },
                "close_friends_member_ids": {
                    "value": close_members,
                    "source_field": field_path_for_persona(
                        persona_id, "social_network.close_friends.members"
                    ),
                },
            },
        }

    return {
        "pair_id": pair.pair_id,
        "selection_reasons": pair.selection_reasons,
        "priority_score": pair.priority_score,
        "source_fields": pair.source_fields,
        "persona_a": serialize_persona(left_id, right_id),
        "persona_b": serialize_persona(right_id, left_id),
        "pair_context": {
            "shared_groups": pair.shared_groups,
            "home_locations": {
                left_id: left_persona.demographics.location_type,
                right_id: right_persona.demographics.location_type,
            },
            "source_fields": pair.source_fields,
        },
    }


def create_evaluation_plan(
    world: SocialWorld,
    *,
    max_pairs: int,
    max_shared_group_size: int,
    include_group_only_pairs: bool,
) -> EvaluationPlan:
    people_by_id = world.people_by_id
    pair_records: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "selection_reasons": set(),
            "priority_score": 0,
            "source_fields": set(),
            "network_layers": {},
            "shared_groups": [],
            "directed_activities": defaultdict(list),
            "directed_conflicts": defaultdict(list),
        }
    )

    for persona in world.people:
        for layer_name in ("intimate_circle", "close_friends"):
            layer = persona.social_network.get(layer_name)
            if not layer:
                continue
            for other_id in layer.members:
                if other_id not in people_by_id or other_id == persona.persona_id:
                    continue
                pair_key = canonical_pair(persona.persona_id, other_id)
                pair_records[pair_key]["network_layers"][persona.persona_id] = layer_name
                pair_records[pair_key]["source_fields"].add(
                    field_path_for_persona(persona.persona_id, "social_network")
                )

        for activity_index, activity in enumerate(persona.past_activities):
            for entity_id in activity.involved_entities:
                if entity_id not in people_by_id:
                    continue
                pair_key = canonical_pair(persona.persona_id, entity_id)
                pair_records[pair_key]["selection_reasons"].add("direct_activity_reference")
                pair_records[pair_key]["priority_score"] += SELECTION_REASON_WEIGHTS[
                    "direct_activity_reference"
                ]
                pair_records[pair_key]["source_fields"].add(
                    field_path_for_persona(
                        persona.persona_id,
                        f"hidden_context_ground_truth.past_activities[{activity_index}]",
                    )
                )
                pair_records[pair_key]["directed_activities"][persona.persona_id].append(
                    _activity_summary(persona.persona_id, activity, activity_index)
                )

        for fact_index, fact in enumerate(persona.static_facts):
            if fact.category != "Taboo-Conflict":
                continue
            match = USER_ID_PATTERN.search(fact.content)
            if not match:
                continue
            target_id = match.group(1)
            if target_id not in people_by_id or target_id == persona.persona_id:
                continue
            pair_key = canonical_pair(persona.persona_id, target_id)
            pair_records[pair_key]["selection_reasons"].add("conflict_fact_reference")
            pair_records[pair_key]["priority_score"] += SELECTION_REASON_WEIGHTS[
                "conflict_fact_reference"
            ]
            pair_records[pair_key]["source_fields"].add(
                field_path_for_persona(
                    persona.persona_id,
                    f"hidden_context_ground_truth.static_facts[{fact_index}]",
                )
            )
            pair_records[pair_key]["directed_conflicts"][persona.persona_id].append(
                _conflict_summary(persona.persona_id, fact, fact_index, target_id)
            )

    for left_id, right_id in list(pair_records):
        left_persona = people_by_id[left_id]
        right_persona = people_by_id[right_id]
        left_layer = left_persona.relationship_layer_for(right_id)
        right_layer = right_persona.relationship_layer_for(left_id)
        pair_records[(left_id, right_id)]["network_layers"] = {
            left_id: left_layer,
            right_id: right_layer,
        }
        if left_layer == "intimate_circle" and right_layer == "intimate_circle":
            pair_records[(left_id, right_id)]["selection_reasons"].add("mutual_intimate_circle")
            pair_records[(left_id, right_id)]["priority_score"] += SELECTION_REASON_WEIGHTS[
                "mutual_intimate_circle"
            ]
        elif left_layer == "close_friends" and right_layer == "close_friends":
            pair_records[(left_id, right_id)]["selection_reasons"].add("mutual_close_friends")
            pair_records[(left_id, right_id)]["priority_score"] += SELECTION_REASON_WEIGHTS[
                "mutual_close_friends"
            ]
        elif left_layer == "intimate_circle" or right_layer == "intimate_circle":
            pair_records[(left_id, right_id)]["selection_reasons"].add("one_sided_intimate_circle")
            pair_records[(left_id, right_id)]["priority_score"] += SELECTION_REASON_WEIGHTS[
                "one_sided_intimate_circle"
            ]
        elif left_layer == "close_friends" or right_layer == "close_friends":
            pair_records[(left_id, right_id)]["selection_reasons"].add("one_sided_close_friends")
            pair_records[(left_id, right_id)]["priority_score"] += SELECTION_REASON_WEIGHTS[
                "one_sided_close_friends"
            ]

    shared_groups_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for group in world.social_groups:
        group_entry = {
            "group_id": group.group_id,
            "group_name": group.group_name,
            "group_type": group.group_type,
            "member_count": len(group.members),
            "source_field": field_path_for_group(group.group_id),
        }
        for left_id, right_id in itertools.combinations(sorted(group.members), 2):
            if left_id not in people_by_id or right_id not in people_by_id:
                continue
            shared_groups_by_pair[(left_id, right_id)].append(group_entry)

    for pair_key, shared_groups in shared_groups_by_pair.items():
        eligible = (
            len(shared_groups) >= 2
            or any(group["member_count"] <= max_shared_group_size for group in shared_groups)
            or pair_key in pair_records
        )
        if not eligible or (pair_key not in pair_records and not include_group_only_pairs):
            continue

        pair_records[pair_key]["shared_groups"] = sorted(
            shared_groups,
            key=lambda item: (item["member_count"], item["group_id"]),
        )
        for group in shared_groups:
            pair_records[pair_key]["source_fields"].add(group["source_field"])

        if len(shared_groups) >= 2:
            pair_records[pair_key]["selection_reasons"].add("shared_multiple_groups")
            pair_records[pair_key]["priority_score"] += SELECTION_REASON_WEIGHTS[
                "shared_multiple_groups"
            ]
        elif any(group["member_count"] <= max_shared_group_size for group in shared_groups):
            pair_records[pair_key]["selection_reasons"].add("shared_small_group")
            pair_records[pair_key]["priority_score"] += SELECTION_REASON_WEIGHTS[
                "shared_small_group"
            ]

    candidate_pairs: list[CandidatePair] = []
    for left_id, right_id in sorted(pair_records):
        record = pair_records[(left_id, right_id)]
        if not record["selection_reasons"]:
            continue
        candidate_pairs.append(
            CandidatePair(
                pair_id=pair_id_for(left_id, right_id),
                personas=(left_id, right_id),
                selection_reasons=sorted(record["selection_reasons"]),
                priority_score=record["priority_score"],
                source_fields=sorted(record["source_fields"]),
                network_layers=record["network_layers"],
                shared_groups=record["shared_groups"],
                directed_activities={
                    left_id: sorted(
                        record["directed_activities"].get(left_id, []),
                        key=lambda item: item["timestamp"],
                    ),
                    right_id: sorted(
                        record["directed_activities"].get(right_id, []),
                        key=lambda item: item["timestamp"],
                    ),
                },
                directed_conflicts={
                    left_id: record["directed_conflicts"].get(left_id, []),
                    right_id: record["directed_conflicts"].get(right_id, []),
                },
            )
        )

    candidate_pairs.sort(key=lambda pair: (-pair.priority_score, pair.pair_id))
    selected_pairs = candidate_pairs[:max_pairs]

    reason_counter = Counter(
        reason for pair in candidate_pairs for reason in pair.selection_reasons
    )
    selection_summary = {
        "candidate_pair_count": len(candidate_pairs),
        "selected_pair_count": len(selected_pairs),
        "max_pairs": max_pairs,
        "reason_counts": dict(sorted(reason_counter.items())),
        "selection_policy": [
            "Pairs are selected from strong social-network ties, direct activities, conflict references, and compact/shared groups.",
            "Priority ranking is only for triage and batching; it does not assign Pass, Warning, or Fail.",
            f"Group-only pairing is limited to pairs from groups of size <= {max_shared_group_size} unless the pair shares multiple groups or already has another signal.",
        ],
    }

    return EvaluationPlan(
        candidate_pairs=candidate_pairs,
        selected_pairs=selected_pairs,
        selection_summary=selection_summary,
    )


def build_request_payload(world: SocialWorld, pair: CandidatePair) -> dict[str, Any]:
    return _build_pair_payload(world, pair)
