from enum import Enum
class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    @classmethod
    def get_all_actions(cls):
        return [Action.UP , Action.DOWN , Action.LEFT , Action.RIGHT]
    
import numpy as np
class SARSAAgent:
    def __init__(self , start_position:np.ndarray , world_map_size:tuple , epsilon = 0.05 , alpha = 0.1 , discount_rate = 1) -> None:
        self.position = start_position
        self.q:np.ndarray = np.zeros(tuple(list(world_map_size) + [4]))
        self.epsilon = epsilon
        self.alpha = alpha
        self.discount_rate = discount_rate
        self.action = -1

    def act(self) -> int:
        if np.random.uniform() < self.epsilon:
            return Action.get_all_actions()[np.random.randint(low=0,high=len(Action.get_all_actions()))]
        x , y = int(self.position[0]) , int(self.position[1])
        indices = np.where(self.q[x,y] == self.q[ x , y].max())[0]
        action_index = indices[0][0] if len(indices) == 1 else indices[np.random.randint(0 , len(indices))]
        return Action.get_all_actions()[action_index]
            
    
    def step(self, reward:int , new_position:np.ndarray):
        new_action = self.act()
        x_0 , y_0 = int(self.position[0]) , int(self.position[1])
        x_1 , y_1 = int(new_position[0]) , int(new_position[1])
        self.q[ x_0,y_0 , self.action.value] += self.alpha * (
            reward + self.discount_rate * (self.q[x_1, y_1, self.action.value] - self.q[x_0,y_0, new_action.value]))
        self.position = new_position
        self.action = new_action

