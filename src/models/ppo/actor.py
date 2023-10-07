from torch.distributions import Categorical
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential, Softmax , Dropout1d
from torch import Tensor
import torch
from ..base_model import BaseModel


class Actor(BaseModel):
    def __init__(self, state_size: int, action_size: int, device="cpu") -> None:
        super().__init__(state_size, action_size, 1, device)
        self.network = Sequential(
            # BatchNorm1d(state_size) ,

            Linear(state_size , 128) , 
            ReLU() , 
            # Dropout1d(p=0.1) , 
            BatchNorm1d(128) ,
            Linear(128 , 256) ,  
            ReLU() , 
            # Dropout1d(p=0.1) ,
            BatchNorm1d(256) , 
            Linear(256 , action_size) , 
            Softmax(dim=-1)
        )
        self.to(device)
        
    def forward(self, state:Tensor) -> Categorical:
        if self.training:
            state = state + 0.1 * torch.randn(size=state.shape)
        actions =  self.network(state)
        return Categorical(actions)