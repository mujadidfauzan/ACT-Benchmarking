"""Episode-local causal history for closed-loop policy inference."""

from collections import deque

import torch


class PolicyHistory:
    """Call observe before predicting, then record the action actually executed."""

    def __init__(self, history_size, rgb_history=False):
        if history_size < 2:
            raise ValueError("history_size must be at least two")
        self.history_size = history_size
        self.rgb_history = rgb_history
        self.proprio = deque(maxlen=history_size)
        self.actions = deque(maxlen=history_size - 1)
        self.images = {}

    @staticmethod
    def _pad(queue, length, reference):
        values = reference.new_zeros((1, length, *reference.shape))
        mask = torch.zeros((1, length), dtype=torch.bool, device=reference.device)
        if queue:
            values[0, -len(queue):] = torch.stack(tuple(queue))
            mask[0, -len(queue):] = True
        return values, mask

    def observe(self, camera_inputs, proprio, normalizer):
        self.proprio.append(proprio[0].detach().clone())
        proprio_values, proprio_mask = self._pad(
            self.proprio, self.history_size, proprio[0]
        )
        action_values, action_mask = self._pad(
            self.actions, self.history_size - 1,
            proprio.new_zeros(normalizer.action_dim),
        )
        inputs = {
            "proprio_history": normalizer.normalize_proprio(proprio_values).masked_fill(
                ~proprio_mask.unsqueeze(-1), 0
            ),
            "action_history": normalizer.normalize_action(action_values).masked_fill(
                ~action_mask.unsqueeze(-1), 0
            ),
            "proprio_history_mask": proprio_mask,
            "action_history_mask": action_mask,
        }
        if self.rgb_history:
            for key, image in camera_inputs.items():
                queue = self.images.setdefault(key, deque(maxlen=self.history_size))
                queue.append(image[0].detach().clone())
                inputs[f"{key}_history"], _ = self._pad(queue, self.history_size, image[0])
        else:
            inputs.update(camera_inputs)
        return inputs

    def record_action(self, action):
        self.actions.append(action.detach().clone())
