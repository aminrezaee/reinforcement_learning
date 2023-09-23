import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import (BatchNorm1d, BCELoss, L1Loss, Linear, Module, ModuleList,
                      ReLU, Softmax)
from torch.optim import Adam

from action import Action


class BaseModel(Module):
    def __init__(self , state_size:int , action_size:int , update_batch_count:int , batch_size:int=20 , device='cpu') -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.update_batch_count = update_batch_count
        self.batch_size = batch_size
        input_size = int(state_size + action_size)
        self.data:Dict[int, # state
                       Dict[Action, # action
                            Tuple[int , float]]] = {} # next_state , reward
        self.device = device
        self.reward_predictor = ModuleList(modules=[
            BatchNorm1d(num_features=input_size) , 
            Linear(in_features=input_size , out_features= int(input_size/2)) , 
            ReLU() , 
            BatchNorm1d(num_features=int(input_size/2)) , 
            Linear(in_features=int(input_size/2) , out_features= int(input_size/4)) , 
            Linear(in_features=int(input_size/4) , out_features= int(input_size/4)) , 
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
        self.data:Dict[int, # state
                       Dict[Action, # action
                            Tuple[int , float]]] = {} # next_state , reward
        
    def get_one_hot(self, index , length):
        onehot = np.zeros(length).astype(np.int64)
        onehot[index] = 1
        return onehot
    
    def unfold_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[np.ndarray]]:
        state_indices = self.data.keys()
        states = []
        actions = []
        rewards = []
        next_states = []
        total_actions = Action.get_all_actions()
        for index in state_indices:
            for action in total_actions:
                if action in self.data[index].keys():
                    states.append(self.get_one_hot(index , self.state_size))
                    actions.append(self.get_one_hot(action.value , self.action_size))
                    rewards.append(self.data[index][action][1])
                    next_states.append(self.get_one_hot(self.data[index][action][0] , self.state_size))
        return states , actions , rewards , next_states

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
        states , actions , rewards , next_states = self.unfold_data()
        inputs = self.create_inputs(actions , states)
        ground_truth_rewards = Tensor(rewards)[:,None]
        ground_truth_next_states = Tensor([state for state in next_states])
        for i in range(self.update_batch_count):
            indices = torch.randperm(len(inputs))[:self.batch_size]
            batch_inputs = inputs#[indices]
            batch_ground_truth_rewards = ground_truth_rewards#[indices]
            batch_ground_truth_next_states = ground_truth_next_states#[indices]
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
    
    def predict(self, batch_actions:List[np.ndarray] , batch_states:List[np.ndarray]) -> Tuple[Tensor , Tensor]:
        self.eval()
        with torch.no_grad():
            inputs = self.create_inputs(batch_actions , batch_states)
            return self._forward(inputs)
    
    def create_inputs(self , batch_actions:List[np.ndarray] , batch_states:List[np.ndarray]) -> Tensor:
        data = np.concatenate((batch_states , batch_actions) , axis=1)
        inputs = Tensor(data , device=self.device)
        return inputs
