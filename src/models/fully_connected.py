from torch.nn import Module , ModuleList , Linear , ReLU , Softmax
from torch import Tensor
from action import Action
import numpy as np
from typing import List , Callable
from torch.nn import MSELoss , BCELoss
class BaseModel(Module):
    def __init__(self , state_size:int , action_size , device='cpu') -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
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

    def reset(self):
        self.states:List[np.ndarray] = [] # state
        self.actions:List[Action] = [] # action 
        self.rewards:List[float] = [] # reward
        self.next_states:List[np.ndarray] = [] # next_states

    def _forward(self, inputs:Tensor):
        rewards = inputs.clone()
        for layer in self.reward_predictor:
            rewards = layer(rewards)
        states = inputs.clone()
        for layer in self.next_state_predictor:
            states = layer(states)
        return rewards , states

    def compute_loss(self) -> Tensor:
        self.train()
        inputs = self.create_inputs(self.actions , self.states)
        ground_truth_rewards = Tensor(self.rewards)
        ground_truth_next_states = Tensor([state for state in self.next_states])
        reward_predictions , state_predictions = self._forward(inputs)
        reward_loss = MSELoss()(reward_predictions , ground_truth_rewards)
        next_state_loss = BCELoss() (state_predictions , ground_truth_next_states)
        return reward_loss + next_state_loss
    
    def predict(self, batch_actions:List[Action] , batch_states:List[np.ndarray]):
        self.eval()
        inputs = self.create_inputs(batch_actions , batch_states)
        return self.decider(inputs)
    
    def create_inputs(self , batch_actions:List[np.ndarray] , batch_states:List[np.ndarray]) -> Tensor:
        data = np.concatenate((batch_states , batch_actions) , axis=1)
        inputs = Tensor(data , device=self.device)
        return inputs
