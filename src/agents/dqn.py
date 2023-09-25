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
    
    def append_observation(self, state:np.ndarray , action:Action , reward:float , next_state:np.ndarray):
        state_index = int(state[0] * len(self.q[0]) + state[1])
        if state_index not in self.model.data.keys():
            self.model.data[state_index] = {}
        self.model.data[state_index][action] = (int(next_state[0] * len(self.q[0]) + next_state[1]) , reward)
    
    def step(self , reward:float , new_position:np.ndarray , current_timestep:int):
        x_0 , x_1 , y_0 , y_1 , _ = self._step(new_position , current_timestep)
        # self.q[ y_0 , x_0 , self.action.value] += self.alpha * ( # q(s , a) = q(s , a) + alpha * ( reward + gamma * (q(s' , a') - q(s , a))
        #     reward + self.discount_rate * (self.q[y_1 , x_1].max()) - self.q[y_0 , x_0 , self.action.value])
        current_q_estimation = self.model.update(self.optimizers_dict)
        new_action = self.act(current_timestep)
        self.action = new_action
        return