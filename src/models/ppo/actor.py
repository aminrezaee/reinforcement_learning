from torch.distributions import Categorical
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential, Softmax
from torch import Tensor

from ..base_model import BaseModel


class Actor(BaseModel):
    def __init__(self, state_size: int, action_size: int, device="cpu") -> None:
        super().__init__(state_size, action_size, 1, device)
        self.network = Sequential(
            BatchNorm1d(state_size) ,
            Linear(state_size , 128) , 
            ReLU() , 
            Linear(128 , 256) , 
            ReLU() , 
            Linear(256 , action_size) , 
            Softmax(dim=0)
        )
        self.to(device)
        
    def forward(self, state:Tensor) -> Categorical:
        actions =  self.network(state)
        return Categorical(actions)