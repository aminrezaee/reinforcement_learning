import numpy as np
from .dyna_q import DynaQAgent
from action import Action
from torch.nn import Module
class DQN(DynaQAgent):
   def __init__(self, start_position: np.ndarray, world_map_size: tuple, 
                model: Module, epsilon: float = 0.05, alpha: float = 0.1, 
                discount_rate: float = 1, learning_rate: float = 0.01) -> None:
      super().__init__(start_position, world_map_size, model, epsilon, alpha, discount_rate, learning_rate, 0)