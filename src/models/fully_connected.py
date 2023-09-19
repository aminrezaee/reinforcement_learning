from torch.nn import Module , ModuleList , Linear , ReLU
from torch import Tensor
from agent import Action
import numpy as np
from typing import List
class BaseModel(Module):
    def __init__(self , input_size:int) -> None:
        super().__init__()
        self.states:List[np.ndarray] = [] # state
        self.actions:List[Action] = [] # action 
        self.rewards:List[float] = [] # reward
        self.decider = ModuleList(modules=[
            Linear(in_features=input_size , out_features= int(input_size/2)) , 
            ReLU() , 
            Linear(in_features=int(input_size/2) , out_features= int(input_size/2)) , 
            ReLU() , 
            Linear(in_features=int(input_size/2) , out_features= int(input_size/4)) , 
            ReLU() , 
            Linear(in_features= int(input_size/4) , out_features=1) , 
        ])

    def reset(self):
        self.states:List[np.ndarray] = [] # state
        self.actions:List[Action] = [] # action 
        self.rewards:List[float] = [] # reward

    def update(self):
        self.train()
        inputs = self.create_inputs(self.actions , self.states)
        ground_truth = Tensor([action.value for action in self.actions])
        predictions = self.decider(inputs)
        
        return
    
    def predict(self, batch_actions:List[Action] , batch_states:List[np.ndarray]):
        self.eval()
        inputs = self.create_inputs(batch_actions , batch_states)
        return self.decider(inputs)
    
    def create_inputs(self , batch_actions , batch_states) -> Tensor:

        return
