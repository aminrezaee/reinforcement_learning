import numpy as np
from .dyna_q import DynaQAgent
from action import Action
from torch.nn import Module
class DQNKeywords:
    states = 'states'
    ground_truth_q_values = 'ground_truth_q_values'
    rewards = 'rewards'
    next_states = 'next_states'
    terminal_stat = 'terminal_stat'
    current_timestep = "current_timestep"
class DQN(DynaQAgent):
    def __init__(self, start_position: np.ndarray, world_map_size: tuple, 
                model: Module, optimizers_dict:dict, epsilon: float = 0.05, alpha: float = 0.1, 
                discount_rate: float = 1) -> None:
      super().__init__(start_position, world_map_size, model , optimizers_dict, epsilon, alpha, discount_rate, 0)
    
    def get_q(self) -> np.ndarray:
        input_dict = {DQNKeywords.states:[self.position]}
        current_q_values = self.model.predict(input_dict).numpy()
        return  current_q_values
    
    def append_observation(self, state:np.ndarray , action:Action , reward:float , next_state:np.ndarray , is_terminal:bool , current_timestep:int):
        current_state_index = self.get_state_index(state)
        if is_terminal:
            ground_truth_q_value = reward
        else:
            input_dict = {DQNKeywords.states : [next_state]} # next state data
            ground_truth_q_value = reward + self.discount_rate * (self.model.predict(input_dict)[0][action.value])
        if current_state_index not in self.model.data:
            self.model.data[current_state_index] = {DQNKeywords.states:state}
        self.model.data[current_state_index][action]  = {DQNKeywords.ground_truth_q_values: ground_truth_q_value , 
                                                DQNKeywords.rewards:reward , 
                                                DQNKeywords.next_states:next_state , 
                                                DQNKeywords.terminal_stat:is_terminal , 
                                                DQNKeywords.current_timestep: current_timestep}

    
    def step(self , new_position:np.ndarray , current_timestep:int):
        x_0 , x_1 , y_0 , y_1 , _ = self._step(new_position , current_timestep)
        new_action = self.act(current_timestep , self.model.random)
        self.action = new_action
        return