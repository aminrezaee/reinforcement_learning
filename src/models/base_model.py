from abc import abstractclassmethod
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, Module, ReLU, Sequential

from action import Action


class BaseModel(Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        update_batch_count: int,
        device="cpu",
    ) -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.update_batch_count = update_batch_count
        self.data: Dict[
            int, Dict[Action, Tuple[int, float]]  # state  # action
        ] = {}  # next_state , reward
        self.device = device

    def dense_layer(self, in_features , out_features = None):
        return Sequential(
            BatchNorm1d(num_features=in_features),
            Linear(in_features=in_features, out_features=out_features if out_features is not None else int(in_features / 2)),
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
        self, keys:List[str]
    ) -> Dict[str , List]:
        state_indices = self.data.keys()
        total_actions = Action.get_all_actions()
        output = {
            key: [] for key in keys
        }
        output["states"] = []
        output["actions"] = []
        for index in state_indices:
            for action in total_actions:
                if action in self.data[index].keys():
                    output['states'].append(self.data[index]['states'])
                    output['actions'].append(self.get_one_hot(action.value, self.action_size))
                    for key in keys:
                        output[key].append(self.data[index][action][key])
        return output

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
        self, input_dict:dict) -> dict:
        return NotImplementedError

    def update(self, optimizers_dict: dict , keys:List[str]) -> Tensor:
        self.train()
        input_dict = self.unfold_data(keys)
        inputs = self.create_inputs(input_dict)
        ground_truth = self.create_ground_truth(input_dict)
        self._update(inputs, ground_truth, optimizers_dict)
        return
