import typing
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy: Dict[int, float]
    hidden_state: List[float]

@dataclass
class Turn:
    action: int
    reward: float
    state: np.ndarray
    value: float
    done: bool

@dataclass
class Batch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    policy_probs: np.ndarray


    def __len__(self):
        return self.observations.shape[0]
