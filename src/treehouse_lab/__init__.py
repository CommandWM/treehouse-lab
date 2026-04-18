"""Treehouse Lab."""

__all__ = ["__version__", "ExperimentResult", "TreehouseLabRunner"]
__version__ = "0.9.0"


def __getattr__(name: str):
    if name in {"ExperimentResult", "TreehouseLabRunner"}:
        from treehouse_lab.runner import ExperimentResult, TreehouseLabRunner

        exports = {
            "ExperimentResult": ExperimentResult,
            "TreehouseLabRunner": TreehouseLabRunner,
        }
        return exports[name]
    msg = f"module 'treehouse_lab' has no attribute {name!r}"
    raise AttributeError(msg)
