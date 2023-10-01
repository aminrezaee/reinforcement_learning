from torch.nn import Linear, Sequential, MSELoss
from torch.optim import Adam
from torch import Tensor
import torch
from .base_model import BaseModel
import logging
import numpy as np
from agents.dqn import DQNKeywords

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
        self.q_value_predictor = Sequential(
            self.dense_layer(state_size , out_features=int(4 * state_size)),
            self.dense_layer(int(4 * state_size)),
            self.dense_layer(int(2 * state_size)),
            Linear(state_size , state_size),
            Linear(in_features=state_size, out_features=action_size),
            Linear(in_features=action_size, out_features=action_size),
        )
        self.random = True
        self.gamma = gamma

    def _forward(self, inputs: Tensor):
        return self.q_value_predictor(inputs)

    def _update(self, inputs, ground_truth: dict, optimizers_dict: dict) -> Tensor:
        optimizer: Adam = optimizers_dict["optimizer"]
        self.random = False
        print(f"input size:{len(inputs)}")
        for i in range(self.update_batch_count):
            # indices = torch.randperm(len(inputs))[: min(len(inputs) , self.batch_size)]
            batch_inputs = inputs#[indices]
            batch_ground_truth_q_values = ground_truth["q_values"]#[indices]
            optimizer.zero_grad()
            q_values_predictions = self._forward(batch_inputs)
            q_values_predictions = torch.gather(q_values_predictions , 1 , torch.argmax(ground_truth['actions'] , dim = 1)[:,None])
            loss: Tensor = MSELoss()(q_values_predictions, batch_ground_truth_q_values[:,None])
            loss_text = f"loss:{round(loss.item() , ndigits=3)}"
            logging.getLogger().log(logging.INFO, loss_text)
            loss.backward()
            optimizer.step()
        return loss
    
    def create_ground_truth(self, input_dict:dict) -> dict:
        ground_truth = dict()
        ground_truth['q_values'] = Tensor(input_dict[DQNKeywords.ground_truth_q_values])
        ground_truth['actions'] = Tensor(input_dict['actions'])
        ground_truth['current_timestep'] = Tensor(input_dict['current_timestep'])
        return ground_truth
    
    def create_inputs(self , input_dict:dict) -> Tensor:
        return Tensor(np.array(input_dict[DQNKeywords.states]) , device=self.device)
