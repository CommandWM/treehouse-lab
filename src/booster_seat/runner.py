from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentResult:
    name: str
    metric: float
    promoted: bool
    notes: str = ""


class BoosterSeatRunner:
    """Placeholder runner for the Booster Seat experiment loop.

    The real implementation should own:
    - dataset loading
    - split enforcement
    - baseline training
    - candidate evaluation
    - incumbent promotion
    - experiment journaling
    """

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)

    def run_baseline(self) -> ExperimentResult:
        raise NotImplementedError("Baseline training is not implemented yet.")

    def run_candidate(self, mutation_name: str) -> ExperimentResult:
        raise NotImplementedError("Candidate evaluation is not implemented yet.")
