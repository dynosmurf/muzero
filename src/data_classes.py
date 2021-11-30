import typing
from typing import Dict, List, Optional

class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy: Dict[int, float]
    hidden_state: List[float]


