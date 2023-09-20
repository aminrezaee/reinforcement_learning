from torch.nn import Module , ModuleList , Linear , ReLU , Softmax
from torch import Tensor
from action import Action
import numpy as np
from typing import List
from torch.nn import MSELoss , BCELoss
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
            Linear(in_features=int(input_size/2) , out_features= int(input_size/2)) , 
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
            Softmax()
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

    def update(self , optimizer:Adam) -> Tensor:
        self.train()
        inputs = self.create_inputs(self.actions , self.states)
        ground_truth_rewards = Tensor(self.rewards)
        ground_truth_next_states = Tensor([state for state in self.next_states])
        for i in range(self.update_batch_count):
            indices = torch.randperm(len(inputs))[:self.batch_size]
            batch_inputs = inputs[indices]
            batch_ground_truth_rewards = ground_truth_rewards[indices]
            batch_ground_truth_next_states = ground_truth_next_states[indices]
            optimizer.zero_grad()
            reward_predictions , state_predictions = self._forward(batch_inputs)
            reward_loss:Tensor = MSELoss()(reward_predictions , batch_ground_truth_rewards)
            next_state_loss:Tensor = BCELoss() (state_predictions , batch_ground_truth_next_states)
            loss = reward_loss + next_state_loss
            loss.backward()
            optimizer.step()
            logging.getLogger().log(logging.INFO , f"loss:{loss.item()}")
        return 
    
    def predict(self, batch_actions:List[Action] , batch_states:List[np.ndarray]) -> Tuple[Tensor , Tensor]:
        self.eval()
        inputs = self.create_inputs(batch_actions , batch_states)
        return self.reward_predictor(inputs) , self.next_state_predictor(inputs)
    
    def create_inputs(self , batch_actions:List[np.ndarray] , batch_states:List[np.ndarray]) -> Tensor:
        data = np.concatenate((batch_states , batch_actions) , axis=1)
        inputs = Tensor(data , device=self.device)
        return inputs
