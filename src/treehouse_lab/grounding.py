from __future__ import annotations

from typing import Any


LOCAL_REFERENCE_SCOPE = "bounded_local_reference"

_COMMON_REFERENCES = [
    {
        "title": "Search space bounds",
        "path": "configs/search_space.yaml",
        "reason": "Shows the declared XGBoost parameter and feature-generation bounds that proposals must stay inside.",
    },
    {
        "title": "Autonomous loop contract",
        "path": "docs/autonomous-loop.md",
        "reason": "Defines bounded mutation selection, journaling, and the no-test-set-leakage loop policy.",
    },
]

_MUTATION_REFERENCES: dict[str, list[dict[str, str]]] = {
    "regularization_tighten": [
        {
            "title": "Evaluation policy",
            "path": "docs/evaluation-policy.md",
            "reason": "Frames train-validation gap checks that justify reducing tree capacity.",
        }
    ],
    "learning_rate_tradeoff": [
        {
            "title": "Benchmark comparison guide",
            "path": "docs/benchmarks.md",
            "reason": "Keeps practical benchmark comparisons tied to rerunnable, attributable moves.",
        }
    ],
    "capacity_increase": [
        {
            "title": "Evaluation policy",
            "path": "docs/evaluation-policy.md",
            "reason": "Clarifies readiness checks before a larger ensemble is considered implementation-ready.",
        }
    ],
    "imbalance_adjustment": [
        {
            "title": "Dataset split contract",
            "path": "docs/evaluation-policy.md",
            "reason": "Keeps class-balance handling on the training path without changing split policy or test usage.",
        }
    ],
    "feature_generation_enable": [
        {
            "title": "Feature generation policy",
            "path": "docs/mvp.md",
            "reason": "Documents that generated features must stay capped and train-only.",
        }
    ],
}


def build_proposal_grounding(
    context: Any,
    mutation_type: str,
    *,
    params_override: dict[str, Any] | None = None,
    feature_generation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    diagnosis = _mapping(getattr(context, "diagnosis", {}))
    tags = _diagnosis_tags(diagnosis)
    evidence = _base_evidence(context)
    if mutation_type == "imbalance_adjustment":
        evidence.append(
            {
                "name": "positive_rate",
                "value": round(float(getattr(context, "positive_rate", 0.5)), 4),
                "reason": "Class weighting is only justified when the validation positive rate is far enough from parity.",
            }
        )
    if mutation_type in {"regularization_tighten", "learning_rate_tradeoff", "capacity_increase"}:
        evidence.append(
            {
                "name": "overfit_gap",
                "value": round(float(getattr(context, "overfit_gap", 0.0)), 4),
                "reason": "Tree capacity and learning-rate moves should respond to the train-validation gap.",
            }
        )
    if params_override:
        evidence.append(
            {
                "name": "params_override",
                "value": _jsonable(params_override),
                "reason": "The exact candidate mutation remains reviewable before execution.",
            }
        )
    if feature_generation:
        evidence.append(
            {
                "name": "feature_generation_plan",
                "value": _jsonable(feature_generation),
                "reason": "Feature generation must be capped, explicit, and separate from parameter-only moves.",
            }
        )

    return {
        "scope": LOCAL_REFERENCE_SCOPE,
        "mutation_type": mutation_type,
        "dataset_key": getattr(context, "dataset_key", None),
        "diagnosis_tags": tags,
        "evidence": evidence,
        "references": _references_for_mutation(mutation_type),
        "constraint": (
            "Use only the incumbent metrics, journal history, declared search-space bounds, and listed candidate payload; "
            "do not invent mutations or use the held-out test set as a search target."
        ),
    }


def build_advisor_grounding(context: dict[str, Any]) -> dict[str, Any]:
    recent_entries = list(context.get("recent_entries", []))
    recent_mutations = [entry.get("name") for entry in recent_entries if entry.get("name")]
    proposal_grounding = _next_proposal_grounding(context)
    references = proposal_grounding.get("references") or _references_for_diagnosis(context)
    diagnosis_preview = _mapping(context.get("diagnosis_preview", {}))
    diagnosis = _mapping(context.get("diagnosis", {}))
    return {
        "dataset_key": context.get("dataset_key"),
        "journal_count": context.get("journal_count", len(recent_entries)),
        "recent_mutations": recent_mutations,
        "diagnosis_tag": _mapping(diagnosis_preview.get("diagnosis", {})).get("primary_tag")
        or diagnosis.get("primary_tag"),
        "bounded_references": references,
        "proposal_grounding": _compact_proposal_grounding(proposal_grounding),
    }


def summarize_step_grounding(step_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for step in step_results:
        proposal = step.get("proposal", {}) if isinstance(step, dict) else {}
        grounding = proposal.get("grounding", {}) if isinstance(proposal, dict) else {}
        if not isinstance(grounding, dict) or not grounding:
            continue
        reference_paths = [
            str(reference.get("path"))
            for reference in grounding.get("references", [])
            if isinstance(reference, dict) and reference.get("path")
        ]
        evidence_names = [
            str(evidence.get("name"))
            for evidence in grounding.get("evidence", [])
            if isinstance(evidence, dict) and evidence.get("name")
        ]
        summaries.append(
            {
                "step_index": step.get("step_index"),
                "mutation_type": proposal.get("mutation_type"),
                "scope": grounding.get("scope"),
                "reference_paths": sorted(dict.fromkeys(reference_paths)),
                "evidence_names": sorted(dict.fromkeys(evidence_names)),
            }
        )
    return summaries


def _base_evidence(context: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": "primary_metric",
            "value": getattr(context, "primary_metric", None),
            "reason": "Candidate quality is judged against the configured validation metric.",
        },
        {
            "name": "promote_threshold",
            "value": float(getattr(context, "promote_threshold", 0.0)),
            "reason": "The candidate must clear this promotion bar before it can replace the incumbent.",
        },
    ]


def _references_for_mutation(mutation_type: str) -> list[dict[str, str]]:
    references = [*_COMMON_REFERENCES, *_MUTATION_REFERENCES.get(mutation_type, [])]
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for reference in references:
        path = reference["path"]
        if path in seen:
            continue
        seen.add(path)
        deduped.append(dict(reference))
    return deduped


def _references_for_diagnosis(context: dict[str, Any]) -> list[dict[str, str]]:
    diagnosis_preview = _mapping(context.get("diagnosis_preview", {}))
    proposal = diagnosis_preview.get("next_proposal") or context.get("recommended_proposal") or {}
    mutation_type = proposal.get("mutation_type")
    if mutation_type:
        return _references_for_mutation(str(mutation_type))
    diagnosis = _mapping(diagnosis_preview.get("diagnosis", {})) or _mapping(context.get("diagnosis", {}))
    preferred = diagnosis.get("preferred_mutations", [])
    if preferred:
        return _references_for_mutation(str(preferred[0]))
    return _references_for_mutation("learning_rate_tradeoff")


def _next_proposal_grounding(context: dict[str, Any]) -> dict[str, Any]:
    diagnosis_preview = _mapping(context.get("diagnosis_preview", {}))
    proposal = diagnosis_preview.get("next_proposal") or context.get("recommended_proposal") or {}
    grounding = proposal.get("grounding", {}) if isinstance(proposal, dict) else {}
    return grounding if isinstance(grounding, dict) else {}


def _compact_proposal_grounding(grounding: dict[str, Any]) -> dict[str, Any]:
    if not grounding:
        return {}
    return {
        "scope": grounding.get("scope"),
        "mutation_type": grounding.get("mutation_type"),
        "reference_paths": [
            reference.get("path")
            for reference in grounding.get("references", [])
            if isinstance(reference, dict) and reference.get("path")
        ],
        "evidence_names": [
            evidence.get("name")
            for evidence in grounding.get("evidence", [])
            if isinstance(evidence, dict) and evidence.get("name")
        ],
    }


def _diagnosis_tags(diagnosis: dict[str, Any]) -> list[str]:
    tags = [str(tag) for tag in diagnosis.get("tags", []) if str(tag)]
    primary_tag = str(diagnosis.get("primary_tag", "")).strip()
    if primary_tag:
        tags.append(primary_tag)
    return sorted(dict.fromkeys(tags))


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    return str(value)
