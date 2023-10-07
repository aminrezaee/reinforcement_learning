from torch.nn import BatchNorm1d, Linear, ReLU, Sequential , Dropout1d
import torch
from ..base_model import BaseModel


class Critic(BaseModel):
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
            Linear(256 , 1)
        )
        self.to(device)
    def forward(self, state):
        state = state + 0.1 * torch.randn(size=state.shape)
        return self.network(state)