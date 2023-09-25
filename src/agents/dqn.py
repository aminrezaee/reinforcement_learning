import numpy as np
from .dyna_q import DynaQAgent
from action import Action
from torch.nn import Module
class DQN(DynaQAgent):
    def __init__(self, start_position: np.ndarray, world_map_size: tuple, 
                model: Module, epsilon: float = 0.05, alpha: float = 0.1, 
                discount_rate: float = 1, learning_rate: float = 0.01) -> None:
      super().__init__(start_position, world_map_size, model, epsilon, alpha, discount_rate, learning_rate, 0)
   
    
    def get_q(self) -> np.ndarray:
        input_dict = {'states':[self.position]}
        inputs = self.model.create_inputs(input_dict)
        current_q_values = self.model.predict(inputs).numpy()
        return  current_q_values
    
    def append_observation(self, state:np.ndarray , action:Action , reward:float , next_state:np.ndarray , is_terminal:bool):
        state_index = int(state[0] * len(self.q[0]) + state[1])
        if is_terminal:
            ground_truth_q_value = reward
        else:
            input_dict = {'states' : [state]}
            ground_truth_q_value = reward + self.discount_rate * (self.model.predict(input_dict)[action.value])
        self.model.data[state_index] = [action , ground_truth_q_value]

    
    def step(self , new_position:np.ndarray , current_timestep:int):
        x_0 , x_1 , y_0 , y_1 , _ = self._step(new_position , current_timestep)
        new_action = self.act(current_timestep)
        self.action = new_action
        return