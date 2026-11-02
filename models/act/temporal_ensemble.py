"""Explicit, episode-local action chunk state; zeros are valid predictions."""

from dataclasses import dataclass, field
import math

import torch


@dataclass
class ChunkState:
    time: int = 0
    predictions: list = field(default_factory=list)

    def add(self, actions):
        if actions.ndim != 3 or actions.shape[1] <= 0:
            raise ValueError("Expected action chunk [B,K,A]")
        self.predictions.append((self.time, actions.detach()))

    def current(self, *, ensemble=False, decay=.01):
        if not math.isfinite(decay) or decay < 0:
            raise ValueError("Temporal ensemble decay must be finite and nonnegative")
        self.predictions = [(start, chunk) for start, chunk in self.predictions
                            if start <= self.time < start + chunk.shape[1]]
        if not self.predictions:
            raise RuntimeError("No action chunk covers the current timestep")
        if ensemble:
            actions = torch.stack([chunk[:, self.time - start] for start, chunk in self.predictions])
            ages = actions.new_tensor([self.time - start for start, _ in self.predictions])
            # Favor recent predictions. This convention is explicit, unlike
            # ordering-dependent weights in some implementations of ACT.
            weights = torch.softmax(-decay * ages, dim=0)
            action = (actions * weights[:, None, None]).sum(0)
        else:
            start, chunk = self.predictions[-1]
            action = chunk[:, self.time - start]
        self.time += 1
        return action
