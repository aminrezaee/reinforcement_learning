from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from torch import Tensor

from action import Action


@dataclass
class MemoryItem:
    state: np.ndarray
    log_prob: float
    value: float
    action: Action
    reward: float
    done: bool

@dataclass
class ReplayMemory:
    batch_size: int
    items:List[MemoryItem] = field(default_factory=list)

    def sample(self , device:str):
        indices = torch.randperm(len(self.items))[: min(len(self.items), self.batch_size)]
        return (
            indices,
            Tensor([self.items[i].state for i in indices] , device=device),
            Tensor([self.items[i].action.value for i in indices], device=device),
            Tensor([self.items[i].log_prob for i in indices], device=device),
            Tensor([self.items[i].value for i in indices], device=device),
            Tensor([self.items[i].reward for i in indices], device=device),
            Tensor([self.items[i].done for i in indices], device=device)
        )

    def append(self, state:np.ndarray, log_prob:float, value:float, action:Action, reward:float, done:bool):
        self.items.append(MemoryItem(state , log_prob , value , action , reward , done))
        
    def reset(self):
        self.items:List[MemoryItem] = []