import logging
from logging import getLogger
from typing import Optional, Tuple

import numpy as np

from action import Action


class Agent:
    def __init__(self, start_position:np.ndarray , world_map_size:tuple , epsilon = 0.05 , alpha = 0.1 , discount_rate = 1) -> None:
        logger = getLogger()
        logger.log(logging.DEBUG, "agent_initialization") 
        self.position = start_position
        self.q:np.ndarray = np.zeros(tuple(list(world_map_size) + [4]))
        self.debug_path = "./../outputs/debug.txt"
        self.epsilon = epsilon
        self.alpha = alpha
        self.discount_rate = discount_rate
        self.action:Action = None

    def act(self, current_timestep:int) -> Action:
        if np.random.uniform() < self.epsilon:
            return Action.get_all_actions()[np.random.randint(low=0, high=len(Action.get_all_actions()))]
        x , y = int(self.position[0]) , int(self.position[1])
        q = self.get_q()
        indices = np.where(q[y,x] == q[ y , x].max())[0]
        with open(self.debug_path , "a+") as file:
            file.write(f"timestep:{current_timestep} \n")
            file.write(f"max:{q[ y , x].max()}\n")
            file.write("_".join([str(i) for i in indices]))
            file.write("\n")
            file.flush()
            file.close()
        action_index = indices[0] if len(indices) == 1 else indices[np.random.randint(0 , len(indices))]
        return Action.get_all_actions()[action_index]
    
    def get_q(self) -> np.ndarray:
        return self.q[self.position[1] , self.position[0]]
    
    def _step(self , new_position:np.ndarray , current_timestep:int , choose_next_action:bool = True) -> Tuple[int , int , int , int , Optional[Tuple[Action , None]]]:
        x_0 , y_0 = int(self.position[0]) , int(self.position[1]) # s_0
        x_1 , y_1 = int(new_position[0]) , int(new_position[1]) # s_1
        self.position = new_position
        new_action = None
        if choose_next_action:
            new_action = self.act(current_timestep)
        return x_0 , x_1 , y_0 , y_1 , new_action
    
    def step(self, reward:int , new_position:np.ndarray , current_timestep:int) -> None:
        return NotImplementedError