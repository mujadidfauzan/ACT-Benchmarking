from dataclasses import dataclass
from typing import Any


@dataclass
class PrimitiveResult:
    obs: Any
    success: bool
    stage: str
