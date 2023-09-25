import logging
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, Module, ReLU, Sequential

from action import Action
from abc import abstractclassmethod


class BaseModel(Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        update_batch_count: int,
        batch_size: int = 20,
        device="cpu",
    ) -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.update_batch_count = update_batch_count
        self.batch_size = batch_size
        self.data: Dict[
            int, Dict[Action, Tuple[int, float]]  # state  # action
        ] = {}  # next_state , reward
        self.device = device

    def dense_layer(self, in_features):
        return Sequential(
            BatchNorm1d(num_features=in_features),
            Linear(in_features=in_features, out_features=int(in_features / 2)),
            ReLU(),
        )

    def reset(self) -> None:
        self.data: Dict[
            int, Dict[Action, Tuple[int, float]]  # state  # action
        ] = {}  # next_state , reward

    def get_one_hot(self, index, length):
        onehot = np.zeros(length).astype(np.int64)
        onehot[index] = 1
        return onehot

    def unfold_data(
        self,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[np.ndarray]]:
        state_indices = self.data.keys()
        states = []
        actions = []
        rewards = []
        next_states = []
        total_actions = Action.get_all_actions()
        for index in state_indices:
            for action in total_actions:
                if action in self.data[index].keys():
                    states.append(self.get_one_hot(index, self.state_size))
                    actions.append(self.get_one_hot(action.value, self.action_size))
                    rewards.append(self.data[index][action][1])
                    next_states.append(
                        self.get_one_hot(self.data[index][action][0], self.state_size)
                    )
        return states, actions, rewards, next_states

    def predict(self, input_dict: dict) -> Any:
        self.eval()
        with torch.no_grad():
            inputs = self.create_inputs(input_dict)
            return self._forward(inputs)

    @abstractclassmethod
    def create_inputs(
        self, input_dict: dict
    ) -> Tensor:  # batch_actions:List[np.ndarray] , batch_states:List[np.ndarray]
        # data = np.concatenate((batch_states , batch_actions) , axis=1)
        # inputs = Tensor(data , device=self.device)
        return NotImplementedError  # inputs

    @abstractclassmethod
    def _forward(self, inputs: Tensor):
        return NotImplementedError

    @abstractclassmethod
    def _update(self, inputs: Tensor, ground_truth: dict, optimizers_dict: dict):
        return NotImplementedError

    @abstractclassmethod
    def create_ground_truth(
        self,
        states: List[np.ndarray],
        rewards: List[float],
        next_states: List[np.ndarray],
        terminal_stats: List[bool],
    ) -> dict:
        return NotImplementedError

    def update(self, optimizers_dict: dict) -> Tensor:
        self.train()
        states, actions, rewards, next_states = self.unfold_data()
        input_dict = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
        }
        inputs = self.create_inputs(input_dict)
        ground_truth = self.create_ground_truth(states, actions, rewards, next_states)
        self._update(inputs, ground_truth, optimizers_dict)
        return
