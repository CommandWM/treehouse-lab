from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class FeatureGenerationPlan:
    enabled: bool
    reason: str
    max_new_features: int
    tool: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FeatureGenerationResult:
    applied: bool
    reason: str
    new_feature_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def should_enable_feature_generation(history: list[dict[str, Any]], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None or len(history) < 2:
        return False
    recent_deltas = [step["result"]["comparison_to_incumbent"].get("delta") for step in history[-2:]]
    bounded_deltas = [abs(float(delta)) for delta in recent_deltas if delta is not None]
    if len(bounded_deltas) < 2:
        return False
    return max(bounded_deltas) < 0.001


def build_feature_plan(config: dict[str, Any], enabled: bool) -> FeatureGenerationPlan:
    feature_config = config.get("feature_generation", {})
    if not enabled:
        return FeatureGenerationPlan(
            enabled=False,
            reason="Parameter-only templates have not plateaued yet, so feature generation remains off.",
            max_new_features=int(feature_config.get("max_new_features", 0)),
        )
    tools = feature_config.get("tools", [])
    return FeatureGenerationPlan(
        enabled=True,
        reason="Parameter-only templates plateaued, so a bounded feature-generation branch is now eligible.",
        max_new_features=int(feature_config.get("max_new_features", 50)),
        tool=str(tools[0]) if tools else None,
    )


def run_feature_plan(plan: FeatureGenerationPlan) -> FeatureGenerationResult:
    if not plan.enabled:
        return FeatureGenerationResult(applied=False, reason=plan.reason)
    return FeatureGenerationResult(
        applied=False,
        reason="Feature-generation execution is not implemented yet; only the stage gate and plan scaffolding exist.",
        new_feature_count=0,
    )
