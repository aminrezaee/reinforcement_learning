from typing import List, Tuple

import numpy as np
from torch.optim import Adam

from action import Action
from models.fully_connected import BaseModel

from .q_learning import QLearningAgent


class DynaQAgent(QLearningAgent):
    def __init__(self, start_position:np.ndarray , 
                 world_map_size:tuple , 
                 epsilon:float = 0.05 , 
                 alpha:float = 0.1 , 
                 discount_rate:float = 1 , 
                 learning_rate:float = 1e-2 , 
                 simulated_observation_count:int = 10) -> None:
        super().__init__(start_position, world_map_size, epsilon, alpha , discount_rate)
        self.model = BaseModel(state_size=int(world_map_size[0] * world_map_size[1]) , 
                               action_size= self.q.shape[-1] , 
                               update_batch_count=1, 
                               batch_size=20)
        self.state_optimizer = Adam(self.model.next_state_predictor.parameters() , lr=learning_rate , weight_decay=1e-4)
        self.reward_optimizer = Adam(self.model.reward_predictor.parameters() , lr=learning_rate , weight_decay=1e-5)
        self.simulated_observation_count = simulated_observation_count

    def append_observation(self, state:np.ndarray , action:Action , reward:float , next_state:np.ndarray):
        state_index = int(state[0] * len(self.q[0]) + state[1])
        if state_index not in self.model.data.keys():
            self.model.data[state_index] = {}
        self.model.data[state_index][action] = (int(next_state[0] * len(self.q[0]) + next_state[1]) , reward)
    
    def get_position(self , prediction:np.ndarray):
        index = np.argmax(prediction)
        return  np.array([int(index/self.q.shape[1]) , int(index % self.q.shape[1])])  
    
    def create_simulated_observations(self) -> Tuple[List[np.ndarray] , List[Action] , List[float] , List[np.ndarray]]:
        states , actions , _ , _ = self.model.unfold_data()
        indices = np.random.choice(np.arange(len(states)), self.simulated_observation_count, replace=False)
        states = [states[i] for i in indices]
        actions = [actions[i] for i in indices]
        rewards , next_states = self.model.predict(actions , states)
        states = [self.get_position(states[i]) for i in range(len(states))]
        next_states = [self.get_position(next_states[i].numpy()) for i in range(len(next_states))]
        rewards = rewards.tolist()
        rewards = [reward[0] for reward in rewards]
        return  states , [Action.get_all_actions()[np.argmax(action)] for action in actions] ,  rewards , next_states

