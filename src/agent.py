from enum import Enum
from logging import getLogger
import logging
from typing import Tuple , Optional

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    @classmethod
    def get_all_actions(cls):
        return [Action.UP , Action.DOWN , Action.LEFT , Action.RIGHT]
    
import numpy as np
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
        indices = np.where(self.q[y,x] == self.q[ y , x].max())[0]
        with open(self.debug_path , "a+") as file:
            file.write(f"timestep:{current_timestep} \n")
            file.write(f"max:{self.q[ y , x].max()}\n")
            file.write("_".join([str(i) for i in indices]))
            file.write("\n")
            file.flush()
            file.close()
        action_index = indices[0] if len(indices) == 1 else indices[np.random.randint(0 , len(indices))]
        return Action.get_all_actions()[action_index]
    
    def _act(self , new_position:np.ndarray , current_timestep:int , choose_next_action:bool = True) -> Tuple[int , int , int , int , Optional[Tuple[Action , None]]]:
        x_0 , y_0 = int(self.position[0]) , int(self.position[1]) # s_0
        x_1 , y_1 = int(new_position[0]) , int(new_position[1]) # s_1
        self.position = new_position
        new_action = None
        if choose_next_action:
            new_action = self.act(current_timestep)
        return x_0 , x_1 , y_0 , y_1 , new_action
    
    def step(self, reward:int , new_position:np.ndarray , current_timestep:int) -> None:
        return NotImplementedError
    
class SARSAAgent(Agent):
    
    def step(self, reward:int , new_position:np.ndarray , current_timestep:int) -> None:
        x_0 , x_1 , y_0 , y_1 , new_action = self._act(new_position , current_timestep)
        self.q[ y_0 , x_0 , self.action.value] += self.alpha * ( # q(s , a) = q(s , a) + alpha * ( reward + gamma * (q(s' , a') - q(s , a))
            reward + self.discount_rate * (self.q[y_1 , x_1 , new_action.value]) - self.q[y_0 , x_0 , self.action.value])
        self.action = new_action

class ExpectedSARSA(Agent):
    def step(self, reward: int, new_position: np.ndarray, current_timestep: int) -> None:
        x_0 , x_1 , y_0 , y_1 , _ = self._act(new_position , current_timestep , choose_next_action=False)
        values = self.q[y_1 , x_1]
        action_count = len(values)
        max_value = values.max()
        greedy_action_count = sum(self.q[y_1, x_1] == max_value)
        if greedy_action_count == action_count:
            actions_probabilities = [1/action_count for _ in values]
        else:
            greedy_prob = (1 - self.epsilon * ((action_count-1)/(action_count)))/greedy_action_count
            random_prob = self.epsilon/action_count
            actions_probabilities = [greedy_prob if value == max_value else random_prob for value in values]
        mean_q_value  = (np.array(actions_probabilities) * self.q[y_1 , x_1]).sum()
        self.q[ y_0 , x_0 , self.action.value] += self.alpha * ( # q(s , a) = q(s , a) + alpha * ( reward + gamma * (q(s' , a') - q(s , a))
            reward + self.discount_rate * mean_q_value - self.q[y_0 , x_0 , self.action.value])
        self.action = self.act(current_timestep)
        return 

class QLearningAgent(Agent):

    def step(self, reward:int , new_position:np.ndarray , current_timestep:int) -> None:
        x_0 , x_1 , y_0 , y_1 , _ = self._act(new_position , current_timestep)
        self.q[ y_0 , x_0 , self.action.value] += self.alpha * ( # q(s , a) = q(s , a) + alpha * ( reward + gamma * (q(s' , a') - q(s , a))
            reward + self.discount_rate * (self.q[y_1 , x_1].max()) - self.q[y_0 , x_0 , self.action.value])
        new_action = self.act(current_timestep)
        self.action = new_action
        return    
