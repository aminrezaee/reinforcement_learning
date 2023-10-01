from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

from .action import Action


@dataclass
class MemoryItem:
    state: np.ndarray
    prob: np.ndarray
    value: float
    action: Action
    reward: float
    done: bool

@dataclass
class ReplayMemory:
    batch_size: int
    items:List[MemoryItem] = field(default_factory=list)

    def sample(self):
        indices = torch.randperm(len(self.items))[: min(len(self.items), self.batch_size)]
        return (
            indices,
            [self.items[i].state for i in indices],
            [self.items[i].action.value for i in indices],
            [self.items[i].prob for i in indices],
            [self.items[i].value for i in indices],
            [self.items[i].reward for i in indices],
            [self.items[i].done for i in indices]
        )

    def append(self, state:np.ndarray, prob:np.ndarray, value:float, action:Action, reward:float, done:bool):
        self.states.append(state)
        self.probs.append(prob)
        self.values.append(value)
        self.actions.append(action)
        self.rewards.append(reward) 
        self.dones.append(done)
        
    def reset(self):
        self.states = []
        self.probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []