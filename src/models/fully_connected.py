from torch.nn import Module , ModuleList , Linear , ReLU , Softmax , Sigmoid
from torch import Tensor
from action import Action
import numpy as np
from typing import List
from torch.nn import MSELoss , BCELoss , L1Loss
from torch.optim import Adam
import logging
import torch
from typing import Tuple

class BaseModel(Module):
    def __init__(self , state_size:int , action_size:int , update_batch_count:int , batch_size:int=20 , device='cpu') -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.update_batch_count = update_batch_count
        self.batch_size = batch_size
        input_size = int(state_size + action_size)
        self.states:List[np.ndarray] = [] # state
        self.actions:List[Action] = [] # action 
        self.rewards:List[float] = [] # reward
        self.next_states:List[np.ndarray] = [] # next_states
        self.device = device
        self.reward_predictor = ModuleList(modules=[
            Linear(in_features=input_size , out_features= int(input_size/2)) , 
            ReLU() , 
            Linear(in_features=int(input_size/2) , out_features= int(input_size/4)) , 
            ReLU() , 
            Linear(in_features= int(input_size/4) , out_features=1) , 
        ])
        self.next_state_predictor = ModuleList(modules=[
            Linear(in_features=input_size , out_features= int(input_size/2)) , 
            ReLU() , 
            Linear(in_features=int(input_size/2) , out_features= int(input_size/2)) , 
            ReLU() , 
            Linear(in_features= int(input_size/2) , out_features=state_size) , 
            Softmax(dim=1)
        ])

    def reset(self) -> None:
        self.states:List[np.ndarray] = [] # state
        self.actions:List[Action] = [] # action 
        self.rewards:List[float] = [] # reward
        self.next_states:List[np.ndarray] = [] # next_states

    def _forward(self, inputs:Tensor) -> Tuple[Tensor , Tensor]:
        rewards = inputs.clone()
        for layer in self.reward_predictor:
            rewards = layer(rewards)
        states = inputs.clone()
        for layer in self.next_state_predictor:
            states = layer(states)
        return rewards , states

    def update(self , state_optimizer:Adam , reward_optimizer:Adam) -> Tensor:
        self.train()
        inputs = self.create_inputs(self.actions , self.states)
        ground_truth_rewards = Tensor(self.rewards)
        ground_truth_next_states = Tensor([state for state in self.next_states])
        for i in range(self.update_batch_count):
            indices = torch.randperm(len(inputs))[:self.batch_size]
            batch_inputs = inputs[indices]
            batch_ground_truth_rewards = ground_truth_rewards[indices]
            batch_ground_truth_next_states = ground_truth_next_states[indices]
            state_optimizer.zero_grad()
            reward_optimizer.zero_grad()
            reward_predictions , state_predictions = self._forward(batch_inputs)
            reward_loss:Tensor = L1Loss()(reward_predictions , batch_ground_truth_rewards)
            next_state_loss:Tensor = BCELoss() (state_predictions , batch_ground_truth_next_states)
            loss = reward_loss + next_state_loss
            reward_loss.backward()
            reward_optimizer.step()
            next_state_loss.backward()
            state_optimizer.step()
            loss_text = f"loss:{round(loss.item() , ndigits=3)} reward_loss:{round(reward_loss.item() , ndigits=3)} next_state_loss:{round(next_state_loss.item(),ndigits=3)}"
            logging.getLogger().log(logging.INFO , loss_text)
        return next_state_loss
    
    def predict(self, batch_actions:List[np.ndarray] , batch_states:List[np.ndarray]) -> Tuple[Tensor , Tensor]:
        self.eval()
        with torch.no_grad():
            inputs = self.create_inputs(batch_actions , batch_states)
            return self._forward(inputs)
    
    def create_inputs(self , batch_actions:List[np.ndarray] , batch_states:List[np.ndarray]) -> Tensor:
        data = np.concatenate((batch_states , batch_actions) , axis=1)
        inputs = Tensor(data , device=self.device)
        return inputs
