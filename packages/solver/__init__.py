"""Range solver — CFR/MCCFR training for imperfect-information poker games."""

from packages.solver.cfr import CFRTrainer
from packages.solver.kuhn import KuhnPoker
from packages.solver.holdem_simplified import SimplifiedHoldem, SimplifiedHoldemConfig
from packages.solver.trainer import TrainingOrchestrator, TrainingConfig, RunStatus

__all__ = [
    "CFRTrainer",
    "KuhnPoker",
    "SimplifiedHoldem",
    "TrainingOrchestrator",
    "TrainingConfig",
    "RunStatus",
]
