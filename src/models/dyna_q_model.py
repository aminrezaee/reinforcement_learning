from .base_model import BaseModel
import torch
from torch import Tensor
from torch.nn import L1Loss , BCELoss , Sequential , Linear , BatchNorm1d , ReLU , Softmax
from torch.optim import Adam
from typing import Tuple , List
import logging
import numpy as np
class DynaQModel(BaseModel):
    def __init__(self, state_size: int, action_size: int, update_batch_count: int, batch_size: int = 20, device='cpu') -> None:
        super().__init__(state_size, action_size, update_batch_count, batch_size, device)
        input_size = int(state_size + action_size)
        self.reward_predictor = Sequential(
            self.dense_layer(input_size) , 
            BatchNorm1d(num_features=int(input_size/2)) , 
            Linear(in_features=int(input_size/2) , out_features= int(input_size/4)) , 
            Linear(in_features=int(input_size/4) , out_features= int(input_size/4)) , 
            Linear(in_features= int(input_size/4) , out_features=1) , 
        )
        self.next_state_predictor = Sequential(
            Linear(in_features=input_size , out_features= int(input_size/2)) , 
            ReLU() , 
            Linear(in_features=int(input_size/2) , out_features= int(input_size/2)) , 
            ReLU() , 
            Linear(in_features= int(input_size/2) , out_features=state_size) , 
            Softmax(dim=1)
        )

    def _forward(self, inputs:Tensor) -> Tuple[Tensor , Tensor]:
        rewards = self.reward_predictor(inputs)
        states = self.next_state_predictor(inputs)
        return rewards , states
    
    def create_ground_truth(self, states , rewards , next_states , terminal_stats) -> dict:
        ground_truth_rewards = Tensor(rewards)[:,None]
        ground_truth_next_states = Tensor([state for state in next_states])
        return {
            'ground_truth_rewards':ground_truth_rewards , 
            'ground_truth_next_states':ground_truth_next_states
        }
    
    def create_inputs(self , input_dict:dict) -> Tensor:
        batch_actions:List[np.ndarray] = input_dict['actions']
        batch_states:List[np.ndarray] = input_dict['states']
        data = np.concatenate((batch_states , batch_actions) , axis=1)
        return Tensor(data , device=self.device)

    def _update(self , inputs:Tensor , ground_truth:dict , optimizers_dict:dict) -> Tensor:
        state_optimizer:Adam = optimizers_dict['state_optimizer']
        reward_optimizer:Adam = optimizers_dict['reward_optimizer']
        for i in range(self.update_batch_count):
            batch_inputs = inputs
            batch_ground_truth_rewards = ground_truth['ground_truth_rewards'] 
            batch_ground_truth_next_states = ground_truth['ground_truth_next_states']
            state_optimizer.zero_grad()
            reward_predictions , state_predictions = self._forward(batch_inputs)
            next_state_loss:Tensor = BCELoss() (state_predictions , batch_ground_truth_next_states)
            reward_loss = None
            next_state_loss.backward()
            state_optimizer.step()
            if next_state_loss < 0.02:
                reward_loss:Tensor = L1Loss()(reward_predictions , batch_ground_truth_rewards)
                reward_optimizer.zero_grad()
                reward_loss.backward()
                reward_optimizer.step()
            if reward_loss is None:
                reward_loss = Tensor([0.0])
            loss = reward_loss + next_state_loss
            loss_text = f"loss:{round(loss.item() , ndigits=3)} reward_loss:{round(reward_loss.item() , ndigits=3)} next_state_loss:{round(next_state_loss.item(),ndigits=3)}"
            logging.getLogger().log(logging.INFO , loss_text)
        return reward_loss , next_state_loss
    