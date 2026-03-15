from __future__ import annotations

from collections import Counter
from datetime import timedelta
from typing import Any

from schemas import (
    Finding,
    NETWORK_LAYER_ORDER,
    PairEvidence,
    PairReport,
    Persona,
    SocialWorld,
    WorldFinding,
    normalize_text,
)


ADULT_CODED_SOCIAL_ACTIVITIES = {
    "coffee meetup",
    "house party",
    "group dinner at restaurant",
}


def _make_finding(
    dimension: str,
    label: str,
    issue_type: str,
    message: str,
    evidence: dict[str, Any] | None = None,
) -> Finding:
    return Finding(
        dimension=dimension,
        label=label,
        issue_type=issue_type,
        message=message,
        evidence=evidence or {},
    )


def _person_lookup(world: SocialWorld, persona_id: str) -> Persona:
    return world.people_by_id[persona_id]


def _activities_for(evidence: PairEvidence, persona_id: str) -> list[dict[str, Any]]:
    return evidence.directed_activities.get(persona_id, [])


def _conflicts_for(evidence: PairEvidence, persona_id: str) -> list[dict[str, Any]]:
    return evidence.directed_conflicts.get(persona_id, [])


def _activity_pair_matches(
    left_activity: dict[str, Any],
    right_activity: dict[str, Any],
) -> bool:
    left_content = normalize_text(left_activity["content"])
    right_content = normalize_text(right_activity["content"])
    if left_content and left_content == right_content:
        return True

    if left_activity["event_type"] != right_activity["event_type"]:
        return False

    if left_activity.get("city_region") and right_activity.get("city_region"):
        if left_activity["city_region"] != right_activity["city_region"]:
            return False

    from schemas import parse_timestamp

    left_ts = parse_timestamp(left_activity["timestamp"])
    right_ts = parse_timestamp(right_activity["timestamp"])
    if left_ts and right_ts and abs(left_ts - right_ts) <= timedelta(days=3):
        return True
    return False


def _strong_tie(evidence: PairEvidence, persona_id: str) -> bool:
    return evidence.network_layers.get(persona_id) in {"intimate_circle", "close_friends"}


def _adult_coded_pair_activity(evidence: PairEvidence) -> list[str]:
    flagged: list[str] = []
    for persona_id in evidence.personas:
        for activity in _activities_for(evidence, persona_id):
            content = normalize_text(activity["content"])
            if content in ADULT_CODED_SOCIAL_ACTIVITIES:
                flagged.append(activity["content"])
    return sorted(set(flagged))


def check_reciprocal_relationship_presence(
    world: SocialWorld,
    evidence: PairEvidence,
) -> Finding:
    left_id, right_id = evidence.personas
    left_layer = evidence.network_layers.get(left_id)
    right_layer = evidence.network_layers.get(right_id)
    if left_layer and right_layer:
        return _make_finding(
            "reciprocal_relationship_presence",
            "pass",
            "consistent",
            f"Both personas classify each other in their social network layers ({left_layer} vs {right_layer}).",
            {"network_layers": evidence.network_layers},
        )

    severity = "fail" if evidence.link_reasons else "warning"
    issue_type = "hard_contradiction" if severity == "fail" else "soft_plausibility_concern"
    return _make_finding(
        "reciprocal_relationship_presence",
        severity,
        issue_type,
        "At least one persona is missing the other from their social network despite other cross-persona evidence.",
        {"network_layers": evidence.network_layers, "link_reasons": evidence.link_reasons},
    )


def check_relationship_strength_consistency(
    world: SocialWorld,
    evidence: PairEvidence,
) -> Finding:
    left_id, right_id = evidence.personas
    left_layer = evidence.network_layers.get(left_id)
    right_layer = evidence.network_layers.get(right_id)
    if not left_layer or not right_layer:
        return _make_finding(
            "relationship_strength_consistency",
            "warning",
            "soft_plausibility_concern",
            "Relationship strength could not be compared cleanly because one side is missing a network layer assignment.",
            {"network_layers": evidence.network_layers},
        )

    distance = abs(
        NETWORK_LAYER_ORDER[left_layer] - NETWORK_LAYER_ORDER[right_layer]
    )
    if distance == 0:
        return _make_finding(
            "relationship_strength_consistency",
            "pass",
            "consistent",
            "Both personas place the relationship at the same strength tier.",
            {"network_layers": evidence.network_layers},
        )
    if distance == 1:
        return _make_finding(
            "relationship_strength_consistency",
            "pass",
            "consistent",
            "The relationship strength differs by one tier, which can happen without contradiction.",
            {"network_layers": evidence.network_layers},
        )
    if distance == 2:
        return _make_finding(
            "relationship_strength_consistency",
            "warning",
            "soft_plausibility_concern",
            "The pair differs by two social-network tiers, which is a notable asymmetry but not a hard contradiction by itself.",
            {"network_layers": evidence.network_layers},
        )
    return _make_finding(
        "relationship_strength_consistency",
        "fail",
        "hard_contradiction",
        "One persona places the other in the intimate circle while the other relegates them to the active network.",
        {"network_layers": evidence.network_layers},
    )


def check_age_and_life_stage_plausibility(
    world: SocialWorld,
    evidence: PairEvidence,
) -> Finding:
    left_persona = _person_lookup(world, evidence.personas[0])
    right_persona = _person_lookup(world, evidence.personas[1])
    younger_age = min(left_persona.demographics.age, right_persona.demographics.age)
    older_age = max(left_persona.demographics.age, right_persona.demographics.age)
    adult_coded_activities = _adult_coded_pair_activity(evidence)

    if younger_age <= 12 and older_age >= 18 and adult_coded_activities:
        return _make_finding(
            "age_and_life_stage_plausibility",
            "fail",
            "hard_contradiction",
            "The pair includes a child linked to clearly adult-coded shared social activity.",
            {
                "ages": {
                    evidence.personas[0]: left_persona.demographics.age,
                    evidence.personas[1]: right_persona.demographics.age,
                },
                "adult_coded_activities": adult_coded_activities,
            },
        )

    if younger_age <= 5 and older_age >= 18:
        child_id = (
            left_persona.persona_id
            if left_persona.demographics.age == younger_age
            else right_persona.persona_id
        )
        child_layer = evidence.network_layers.get(child_id)
        if child_layer == "close_friends":
            return _make_finding(
                "age_and_life_stage_plausibility",
                "fail",
                "hard_contradiction",
                "A toddler or preschooler is classifying an adult as a close friend rather than primarily as a caregiver/intimate tie.",
                {
                    "ages": {
                        evidence.personas[0]: left_persona.demographics.age,
                        evidence.personas[1]: right_persona.demographics.age,
                    },
                    "network_layers": evidence.network_layers,
                },
            )

    if younger_age < 18 and older_age >= 18 and evidence.age_gap >= 25 and (
        _strong_tie(evidence, evidence.personas[0]) or _strong_tie(evidence, evidence.personas[1])
    ):
        return _make_finding(
            "age_and_life_stage_plausibility",
            "warning",
            "soft_plausibility_concern",
            "The pair has a large adult-minor age gap inside a strong social tier and should be interpreted carefully as family, caregiving, or mentorship rather than peer friendship.",
            {
                "ages": {
                    evidence.personas[0]: left_persona.demographics.age,
                    evidence.personas[1]: right_persona.demographics.age,
                },
                "network_layers": evidence.network_layers,
            },
        )

    return _make_finding(
        "age_and_life_stage_plausibility",
        "pass",
        "consistent",
        "No age or life-stage contradiction is directly implied by the available pair evidence.",
        {
            "ages": {
                evidence.personas[0]: left_persona.demographics.age,
                evidence.personas[1]: right_persona.demographics.age,
            }
        },
    )


def check_child_dependency_realism(
    world: SocialWorld,
    evidence: PairEvidence,
) -> Finding:
    left_persona = _person_lookup(world, evidence.personas[0])
    right_persona = _person_lookup(world, evidence.personas[1])
    minor_persona = None
    adult_persona = None
    if left_persona.demographics.age < 18 <= right_persona.demographics.age:
        minor_persona, adult_persona = left_persona, right_persona
    elif right_persona.demographics.age < 18 <= left_persona.demographics.age:
        minor_persona, adult_persona = right_persona, left_persona

    if not minor_persona or not adult_persona:
        return _make_finding(
            "child_dependency_realism",
            "pass",
            "consistent",
            "The pair is not an adult-child relationship, so caregiver realism is not directly at issue.",
        )

    adult_intimates = [
        member_id
        for member_id in minor_persona.social_network.get("intimate_circle", []).members
        if _person_lookup(world, member_id).demographics.age >= 18
    ]
    if adult_persona.persona_id in adult_intimates:
        return _make_finding(
            "child_dependency_realism",
            "pass",
            "consistent",
            "The adult appears in the minor's intimate circle, which is consistent with a caregiver or close family tie.",
            {
                "minor_id": minor_persona.persona_id,
                "adult_id": adult_persona.persona_id,
                "adult_intimates_for_minor": adult_intimates,
            },
        )

    if minor_persona.demographics.age <= 5 and _strong_tie(evidence, minor_persona.persona_id):
        return _make_finding(
            "child_dependency_realism",
            "fail",
            "hard_contradiction",
            "A toddler or preschooler has a strong adult tie without that adult appearing in the child's intimate caregiver-like circle.",
            {
                "minor_id": minor_persona.persona_id,
                "adult_id": adult_persona.persona_id,
                "network_layers": evidence.network_layers,
                "adult_intimates_for_minor": adult_intimates,
            },
        )

    if _strong_tie(evidence, minor_persona.persona_id) or _strong_tie(evidence, adult_persona.persona_id):
        return _make_finding(
            "child_dependency_realism",
            "warning",
            "soft_plausibility_concern",
            "The pair has a strong adult-child tie, but the adult is not visible in the child's intimate circle.",
            {
                "minor_id": minor_persona.persona_id,
                "adult_id": adult_persona.persona_id,
                "network_layers": evidence.network_layers,
                "adult_intimates_for_minor": adult_intimates,
            },
        )

    return _make_finding(
        "child_dependency_realism",
        "pass",
        "consistent",
        "The adult-child tie is weak enough that missing caregiver evidence is not a contradiction for this pair alone.",
        {
            "minor_id": minor_persona.persona_id,
            "adult_id": adult_persona.persona_id,
            "network_layers": evidence.network_layers,
        },
    )


def check_location_compatibility(
    world: SocialWorld,
    evidence: PairEvidence,
) -> Finding:
    left_persona = _person_lookup(world, evidence.personas[0])
    right_persona = _person_lookup(world, evidence.personas[1])
    if evidence.shared_city:
        return _make_finding(
            "location_compatibility",
            "pass",
            "consistent",
            "Both personas are anchored to the same home city/region.",
            {
                left_persona.persona_id: left_persona.demographics.city_region,
                right_persona.persona_id: right_persona.demographics.city_region,
            },
        )

    direct_activities = _activities_for(evidence, evidence.personas[0]) + _activities_for(
        evidence, evidence.personas[1]
    )
    in_person_activities = [activity for activity in direct_activities if not activity["is_remote"]]
    if in_person_activities:
        return _make_finding(
            "location_compatibility",
            "warning",
            "soft_plausibility_concern",
            "The personas live in different cities but are linked by in-person activity evidence, so travel or mirrored context should be checked.",
            {
                "home_cities": {
                    left_persona.persona_id: left_persona.demographics.city_region,
                    right_persona.persona_id: right_persona.demographics.city_region,
                },
                "in_person_activities": in_person_activities,
            },
        )

    if _strong_tie(evidence, evidence.personas[0]) or _strong_tie(evidence, evidence.personas[1]):
        return _make_finding(
            "location_compatibility",
            "warning",
            "soft_plausibility_concern",
            "The pair has a strong tie across different home cities; this is plausible but should be interpreted as long-distance family/friendship unless more local evidence exists.",
            {
                "home_cities": {
                    left_persona.persona_id: left_persona.demographics.city_region,
                    right_persona.persona_id: right_persona.demographics.city_region,
                }
            },
        )

    return _make_finding(
        "location_compatibility",
        "pass",
        "consistent",
        "No location contradiction is implied by the available pair evidence.",
        {
            "home_cities": {
                left_persona.persona_id: left_persona.demographics.city_region,
                right_persona.persona_id: right_persona.demographics.city_region,
            }
        },
    )


def check_shared_activity_consistency(
    world: SocialWorld,
    evidence: PairEvidence,
) -> Finding:
    left_id, right_id = evidence.personas
    left_activities = _activities_for(evidence, left_id)
    right_activities = _activities_for(evidence, right_id)
    if left_activities and right_activities:
        if any(
            _activity_pair_matches(left_activity, right_activity)
            for left_activity in left_activities
            for right_activity in right_activities
        ):
            return _make_finding(
                "shared_activity_consistency",
                "pass",
                "consistent",
                "Both personas contain compatible activity evidence referencing each other.",
                {"left_activities": left_activities, "right_activities": right_activities},
            )
        return _make_finding(
            "shared_activity_consistency",
            "warning",
            "soft_plausibility_concern",
            "Both personas reference each other in activities, but there is no clearly mirrored event pair. This may reflect separate interactions rather than a contradiction.",
            {"left_activities": left_activities, "right_activities": right_activities},
        )

    if left_activities or right_activities:
        return _make_finding(
            "shared_activity_consistency",
            "warning",
            "missing_mirrored_evidence",
            "Only one side mentions the shared activity, so mirrored evidence is missing.",
            {"left_activities": left_activities, "right_activities": right_activities},
        )

    return _make_finding(
        "shared_activity_consistency",
        "pass",
        "consistent",
        "No direct shared activity evidence exists for this pair, so there is nothing contradictory to reconcile.",
    )


def check_conflict_consistency(
    world: SocialWorld,
    evidence: PairEvidence,
) -> Finding:
    left_id, right_id = evidence.personas
    left_conflicts = _conflicts_for(evidence, left_id)
    right_conflicts = _conflicts_for(evidence, right_id)
    if left_conflicts and right_conflicts:
        left_reasons = {normalize_text(item["content"]) for item in left_conflicts}
        right_reasons = {normalize_text(item["content"]) for item in right_conflicts}
        if left_reasons == right_reasons:
            if evidence.network_layers.get(left_id) == "intimate_circle" or evidence.network_layers.get(right_id) == "intimate_circle":
                return _make_finding(
                    "conflict_consistency",
                    "fail",
                    "hard_contradiction",
                    "The pair mirrors a conflict fact but still places each other in the intimate circle.",
                    {
                        "network_layers": evidence.network_layers,
                        "left_conflicts": left_conflicts,
                        "right_conflicts": right_conflicts,
                    },
                )
            if _strong_tie(evidence, left_id) or _strong_tie(evidence, right_id):
                return _make_finding(
                    "conflict_consistency",
                    "warning",
                    "soft_plausibility_concern",
                    "The conflict is mirrored, but the pair is still maintained as a strong tie.",
                    {
                        "network_layers": evidence.network_layers,
                        "left_conflicts": left_conflicts,
                        "right_conflicts": right_conflicts,
                    },
                )
            return _make_finding(
                "conflict_consistency",
                "pass",
                "consistent",
                "Both personas mirror the conflict fact with compatible wording.",
                {"left_conflicts": left_conflicts, "right_conflicts": right_conflicts},
            )
        return _make_finding(
            "conflict_consistency",
            "fail",
            "hard_contradiction",
            "Both personas mention a conflict, but the conflict descriptions do not agree.",
            {"left_conflicts": left_conflicts, "right_conflicts": right_conflicts},
        )

    if left_conflicts or right_conflicts:
        return _make_finding(
            "conflict_consistency",
            "warning",
            "missing_mirrored_evidence",
            "A conflict fact exists on only one side of the pair.",
            {"left_conflicts": left_conflicts, "right_conflicts": right_conflicts},
        )

    return _make_finding(
        "conflict_consistency",
        "pass",
        "consistent",
        "No conflict evidence exists for this pair.",
    )


def check_marital_household_plausibility(
    world: SocialWorld,
    evidence: PairEvidence,
) -> Finding:
    left_persona = _person_lookup(world, evidence.personas[0])
    right_persona = _person_lookup(world, evidence.personas[1])
    minors_with_marital_status = [
        persona.persona_id
        for persona in (left_persona, right_persona)
        if persona.demographics.age < 18 and persona.demographics.marital_status != "N/A"
    ]
    if minors_with_marital_status:
        return _make_finding(
            "marital_household_plausibility",
            "fail",
            "hard_contradiction",
            "A minor has a non-N/A marital status.",
            {"persona_ids": minors_with_marital_status},
        )

    if (
        evidence.same_last_name
        and evidence.shared_city
        and evidence.network_layers.get(left_persona.persona_id) == "intimate_circle"
        and evidence.network_layers.get(right_persona.persona_id) == "intimate_circle"
    ):
        return _make_finding(
            "marital_household_plausibility",
            "pass",
            "consistent",
            "Shared surname, shared city, and reciprocal intimate-circle placement are compatible with a household or close family tie.",
            {
                "marital_statuses": {
                    left_persona.persona_id: left_persona.demographics.marital_status,
                    right_persona.persona_id: right_persona.demographics.marital_status,
                }
            },
        )

    return _make_finding(
        "marital_household_plausibility",
        "pass",
        "consistent",
        "The schema has marital_status but no explicit spouse or household identifiers, so no direct household contradiction can be asserted for this pair.",
        {
            "marital_statuses": {
                left_persona.persona_id: left_persona.demographics.marital_status,
                right_persona.persona_id: right_persona.demographics.marital_status,
            }
        },
    )


def check_missing_mirrored_facts(
    world: SocialWorld,
    evidence: PairEvidence,
) -> Finding:
    left_id, right_id = evidence.personas
    left_conflicts = _conflicts_for(evidence, left_id)
    right_conflicts = _conflicts_for(evidence, right_id)
    left_layer = evidence.network_layers.get(left_id)
    right_layer = evidence.network_layers.get(right_id)

    if (left_conflicts and not right_conflicts) or (right_conflicts and not left_conflicts):
        return _make_finding(
            "missing_mirrored_facts",
            "warning",
            "missing_mirrored_evidence",
            "The pair has one-sided conflict evidence without a mirrored mention from the other side.",
            {"left_conflicts": left_conflicts, "right_conflicts": right_conflicts},
        )

    if (
        left_layer
        and right_layer
        and abs(NETWORK_LAYER_ORDER[left_layer] - NETWORK_LAYER_ORDER[right_layer]) >= 2
        and not evidence.shared_groups
        and not _activities_for(evidence, left_id)
        and not _activities_for(evidence, right_id)
        and not left_conflicts
        and not right_conflicts
    ):
        return _make_finding(
            "missing_mirrored_facts",
            "warning",
            "missing_mirrored_evidence",
            "The relationship is strongly asymmetric, and there is no mirrored activity, conflict, or group evidence to explain the difference.",
            {"network_layers": evidence.network_layers},
        )

    return _make_finding(
        "missing_mirrored_facts",
        "pass",
        "consistent",
        "No suspicious missing mirrored facts were detected for this pair.",
    )


def _overall_pair_label(checks: list[Finding]) -> str:
    if any(check.label == "fail" for check in checks):
        return "fail"
    if any(check.label == "warning" for check in checks):
        return "warning"
    return "pass"


def check_overall_social_world_coherence(checks: list[Finding]) -> Finding:
    overall_label = _overall_pair_label(checks)
    if overall_label == "fail":
        return _make_finding(
            "overall_social_world_coherence",
            "fail",
            "hard_contradiction",
            "The pair contains at least one hard contradiction, so the local social-world coherence fails.",
        )
    if overall_label == "warning":
        return _make_finding(
            "overall_social_world_coherence",
            "warning",
            "soft_plausibility_concern",
            "The pair is directionally coherent but contains warnings that merit review.",
        )
    return _make_finding(
        "overall_social_world_coherence",
        "pass",
        "consistent",
        "No pair-level contradictions or plausibility concerns were detected.",
    )


def _serialize_finding(finding: Finding) -> dict[str, Any]:
    return {
        "dimension": finding.dimension,
        "label": finding.label,
        "issue_type": finding.issue_type,
        "message": finding.message,
        "evidence": finding.evidence,
    }


def evaluate_pair(world: SocialWorld, evidence: PairEvidence) -> PairReport:
    checks = [
        check_reciprocal_relationship_presence(world, evidence),
        check_relationship_strength_consistency(world, evidence),
        check_age_and_life_stage_plausibility(world, evidence),
        check_child_dependency_realism(world, evidence),
        check_location_compatibility(world, evidence),
        check_shared_activity_consistency(world, evidence),
        check_conflict_consistency(world, evidence),
        check_marital_household_plausibility(world, evidence),
        check_missing_mirrored_facts(world, evidence),
    ]
    checks.append(check_overall_social_world_coherence(checks))

    overall_label = _overall_pair_label(checks)
    hard_contradictions = [finding.message for finding in checks if finding.issue_type == "hard_contradiction"]
    soft_concerns = [finding.message for finding in checks if finding.issue_type == "soft_plausibility_concern"]
    mirrored_warnings = [finding.message for finding in checks if finding.issue_type == "missing_mirrored_evidence"]

    return PairReport(
        pair_id=evidence.pair_id,
        personas=evidence.personas,
        overall_label=overall_label,
        evidence={
            "network_layers": evidence.network_layers,
            "link_reasons": evidence.link_reasons,
            "shared_groups": evidence.shared_groups,
            "directed_activities": evidence.directed_activities,
            "directed_conflicts": evidence.directed_conflicts,
            "shared_city": evidence.shared_city,
            "same_last_name": evidence.same_last_name,
            "age_gap": evidence.age_gap,
        },
        checks=checks,
        hard_contradictions=hard_contradictions,
        soft_plausibility_concerns=soft_concerns,
        missing_mirrored_evidence=mirrored_warnings,
    )


def evaluate_pairs(world: SocialWorld, candidate_pairs: list[PairEvidence]) -> list[PairReport]:
    return [evaluate_pair(world, evidence) for evidence in candidate_pairs]


def evaluate_world(world: SocialWorld) -> list[WorldFinding]:
    findings: list[WorldFinding] = []
    people_by_id = world.people_by_id

    for persona in world.people:
        seen_members: list[str] = []
        for layer in persona.social_network.values():
            seen_members.extend(layer.members)
        expected_members = set(people_by_id) - {persona.persona_id}
        if set(seen_members) != expected_members or len(seen_members) != len(expected_members):
            findings.append(
                WorldFinding(
                    label="fail",
                    issue_type="hard_contradiction",
                    message="A persona's social network does not cleanly partition the rest of the world.",
                    entity_ids=[persona.persona_id],
                    evidence={
                        "expected_member_count": len(expected_members),
                        "observed_member_count": len(seen_members),
                    },
                )
            )

    for group in world.social_groups:
        missing_members = [member for member in group.members if member not in people_by_id]
        if missing_members:
            findings.append(
                WorldFinding(
                    label="fail",
                    issue_type="hard_contradiction",
                    message="A social group references persona IDs that are not present in the people array.",
                    entity_ids=[group.group_id],
                    evidence={"missing_members": missing_members},
                )
            )

    for persona in world.people:
        if persona.demographics.age >= 13:
            continue
        adult_intimates = [
            member_id
            for member_id in persona.social_network["intimate_circle"].members
            if people_by_id[member_id].demographics.age >= 18
        ]
        if not adult_intimates:
            findings.append(
                WorldFinding(
                    label="fail",
                    issue_type="hard_contradiction",
                    message="A child has no adult in the intimate circle, which undermines caregiver realism in the shared world.",
                    entity_ids=[persona.persona_id],
                    evidence={"age": persona.demographics.age},
                )
            )

        adult_close_friends = [
            member_id
            for member_id in persona.social_network["close_friends"].members
            if people_by_id[member_id].demographics.age >= 18
        ]
        if persona.demographics.age <= 5 and len(adult_close_friends) > 2:
            findings.append(
                WorldFinding(
                    label="fail",
                    issue_type="hard_contradiction",
                    message="A toddler or preschooler has too many adults in the close-friends layer, which reads as an adult-style network rather than a child dependency structure.",
                    entity_ids=[persona.persona_id],
                    evidence={
                        "age": persona.demographics.age,
                        "adult_close_friend_count": len(adult_close_friends),
                        "adult_close_friends": adult_close_friends,
                    },
                )
            )
        elif 6 <= persona.demographics.age <= 12 and len(adult_close_friends) > 5:
            findings.append(
                WorldFinding(
                    label="warning",
                    issue_type="soft_plausibility_concern",
                    message="A child's close-friends layer contains many adults, which may indicate an overly adult-shaped network.",
                    entity_ids=[persona.persona_id],
                    evidence={
                        "age": persona.demographics.age,
                        "adult_close_friend_count": len(adult_close_friends),
                        "adult_close_friends": adult_close_friends,
                    },
                )
            )

    return findings


def summarize_pair_reports(pair_reports: list[PairReport]) -> dict[str, Any]:
    overall_counter = Counter(report.overall_label for report in pair_reports)
    dimension_counter = Counter()
    issue_type_counter = Counter()
    for report in pair_reports:
        for check in report.checks:
            dimension_counter[(check.dimension, check.label)] += 1
            issue_type_counter[check.issue_type] += 1

    return {
        "pair_count": len(pair_reports),
        "overall_label_counts": dict(sorted(overall_counter.items())),
        "dimension_label_counts": {
            f"{dimension}:{label}": count
            for (dimension, label), count in sorted(dimension_counter.items())
        },
        "issue_type_counts": dict(sorted(issue_type_counter.items())),
    }


def pair_report_to_dict(report: PairReport) -> dict[str, Any]:
    return {
        "pair_id": report.pair_id,
        "personas": list(report.personas),
        "overall_label": report.overall_label,
        "evidence": report.evidence,
        "checks": [_serialize_finding(finding) for finding in report.checks],
        "hard_contradictions": report.hard_contradictions,
        "soft_plausibility_concerns": report.soft_plausibility_concerns,
        "missing_mirrored_evidence": report.missing_mirrored_evidence,
    }


def world_finding_to_dict(finding: WorldFinding) -> dict[str, Any]:
    return {
        "label": finding.label,
        "issue_type": finding.issue_type,
        "message": finding.message,
        "entity_ids": finding.entity_ids,
        "evidence": finding.evidence,
    }
