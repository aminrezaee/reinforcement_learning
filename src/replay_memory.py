from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from torch import Tensor
from scipy.signal import lfilter
from action import Action


@dataclass
class MemoryItem:
    state: np.ndarray
    log_prob: float
    value: float
    action: Action
    reward: float
    done: bool
    advantage:int = -1

@dataclass
class ReplayMemory:
    batch_size: int
    items:List[MemoryItem] = field(default_factory=list)

    def sample(self , device:str):
        first_index = torch.randint(low=0 , high=int(len(self.items) - self.batch_size) , size = (1,))
        indices = [int(first_index + i) for i in range(self.batch_size)]
        return (
            indices,
            Tensor([self.items[i].state for i in indices] , device=device),
            Tensor([self.items[i].action.value for i in indices], device=device),
            Tensor([self.items[i].log_prob for i in indices], device=device),
            Tensor([self.items[i].value for i in indices], device=device),
            Tensor([self.items[i].reward for i in indices], device=device),
            Tensor([self.items[i].done for i in indices], device=device) , 
            Tensor([self.items[i].advantage for i in indices], device=device)
        )

    def append(self, state:np.ndarray, log_prob:float, value:float, action:Action, reward:float, done:bool):
        self.items.append(MemoryItem(state , log_prob , value , action , reward , done))

    def get_delta(self, gamma:float , index):
        reward = self.items[index].reward
        next_value = self.items[int(index + 1)]
        value = self.items[index].value
        return reward + gamma * next_value - value
        
    def set_advantages(self , gamma:float , lamda:float):
        rewards = np.array([item.reward for item in self.items])
        values = np.array([item.value for item in self.items])
        deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
        discount_coefficient = gamma * lamda
        advantages = lfilter([1], [1, float(-discount_coefficient)], deltas[::-1], axis=0)[::-1]
        mean = np.mean(advantages , axis=0)
        std = np.std(advantages , axis=0)
        advantages = (advantages - mean)/std

        for i in range(len(advantages)):
            self.items[i].advantage = advantages[i]
        return
        
    def reset(self):
        self.items:List[MemoryItem] = []
    