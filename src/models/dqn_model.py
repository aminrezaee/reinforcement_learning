from torch.nn import Linear, Sequential, MSELoss
from torch.optim import Adam
from torch import Tensor
import torch
from .base_model import BaseModel
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np
from typing import List
def calculate_q_value(gamma , reward , next_state_q_value , current_q_value , terminal_stat):
    if terminal_stat:
        return reward
    return current_q_value + gamma * next_state_q_value

class DQNModel(BaseModel):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        update_batch_count: int,
        batch_size: int = 20,
        device="cpu",
        gamma:float=1.0
    ) -> None:
        super().__init__(
            state_size, action_size, update_batch_count, batch_size, device
        )
        self.reward_predictor = None
        self.next_state_predictor = None
        self.q_value_predictor = Sequential(
            self.dense_layer(state_size),
            self.dense_layer(int(state_size / 2)),
            Linear(in_features=int(state_size / 4), out_features=action_size),
            Linear(in_features=action_size, out_features=action_size),
        )
        self.gamma = gamma

    def _forward(self, inputs: Tensor):
        return self.q_value_predictor(inputs)

    def _update(self, inputs, ground_truth: dict, optimizers_dict: dict) -> Tensor:
        optimizer: Adam = optimizers_dict["optimizer"]

        for i in range(self.update_batch_count):
            indices = torch.randperm(len(inputs))[: self.batch_size]
            batch_inputs = inputs[indices]
            batch_ground_truth_q_values = ground_truth["q_values"][indices]
            optimizer.zero_grad()
            q_values_predictions = self._forward(batch_inputs)
            loss: Tensor = MSELoss()(q_values_predictions, batch_ground_truth_q_values)
            loss_text = f"loss:{round(loss.item() , ndigits=3)}"
            logging.getLogger().log(logging.INFO, loss_text)
            loss.backward()
            optimizer.step()
        return loss
    
    def create_ground_truth(self, states:List[np.ndarray], rewards:List[float], next_states:List[np.ndarray], terminal_stats:List[bool]) -> dict:
        ground_truth = dict()
        current_q_values = self.predict({'states' : states})
        next_states_q_values = self.predict({'states' : next_states})
        calculate_q_value_ = partial(calculate_q_value , self.gamma) 
        with ThreadPoolExecutor(5) as executor:
            q_values = executor.map(calculate_q_value_ , [item for item in zip( rewards , next_states_q_values , current_q_values , terminal_stats)])
        ground_truth['q_values'] = Tensor(q_values)
        return ground_truth
    
    def create_inputs(self , input_dict:dict) -> Tensor:
        return Tensor(np.ndarray(input_dict['states']) , device=self.device)
