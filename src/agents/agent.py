import logging
from logging import getLogger

import numpy as np

from action import Action
from replay_memory import ReplayMemory


class Agent:
    def __init__(
        self,
        start_position: np.ndarray,
        world_map_size: tuple,
        batch_size:int,
        epsilon=0.05,
        alpha=0.1,
        discount_rate=1,
        device:str = "cpu"
    ) -> None:
        logger = getLogger()
        logger.log(logging.DEBUG, "agent_initialization")
        self.position = start_position
        self.q: np.ndarray = np.zeros(tuple(list(world_map_size) + [4]))
        self.debug_path = "./../outputs/debug.txt"
        self.epsilon = epsilon
        self.alpha = alpha
        self.discount_rate = discount_rate
        self.memory = ReplayMemory(batch_size)
        self.action: Action = None
        self.world_map_size = world_map_size
        self.device = device

    def get_q(self) -> np.ndarray:
        return self.q[self.position[1], self.position[0]]

    def get_state_index(self, position: np.ndarray):
        return int(position[0] * self.world_map_size[0] + position[1])

    def step(self, new_position: np.ndarray) -> None: # set position + act
        return NotImplementedError
    

    def epsilon_greedy(self , q:np.ndarray , random=False) -> Action:
        if np.random.uniform() < self.epsilon or random:
            return Action.get_all_actions()[
                np.random.randint(low=0, high=len(Action.get_all_actions()))
            ]
        indices = np.where(q == q.max())[0]
        action_index = (
            indices[0]
            if len(indices) == 1
            else indices[np.random.randint(0, len(indices))]
        )
        return Action.get_all_actions()[action_index]
    
    def learn(self) -> None:
        return NotImplementedError
    
    def save_models(self , path) -> None:
        return NotImplementedError
    
    def append_observation(
        self,
        state: np.ndarray,
        prob: np.ndarray,
        value: float,
        action: Action,
        reward: float,
        done: bool,
    ) -> None:
        self.memory.append(state, prob, value, action, reward, done)
