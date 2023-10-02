import logging
import os

import numpy as np 

from agents.agent import Agent
from environments.base_environment import BaseEnvironment


class BaseTrainer:
    def __init__(self , 
                 agent:Agent , 
                 environment:BaseEnvironment, 
                 maximum_timesteps:int , 
                 verbose:bool = True) -> None:
        self.agent = agent 
        self.environment = environment
        self.maximum_timesteps = maximum_timesteps
        self.current_timestep:int = 0
        os.makedirs(f"{self.environment.output_path}/models" , exist_ok=True)
        self.verbose = verbose
    
    def train(self):
        reward = 0
        reward_sum = 0
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        best_average_return = -np.inf
        while self.current_timestep < self.maximum_timesteps:
            is_done = False
            position = self.environment.reset()
            current_return = 0
            episode_timestep = 0
            while not is_done:
                logging.getLogger().log(logging.INFO ,f"timestep:{self.current_timestep}")
                action , prob , value = self.agent.step(position)
                position , reward , is_done , _ , _ = self.environment.step(self.agent , self.maximum_timesteps)
                self.agent.memory.append(self.agent.position , prob , value , action , reward , is_done)
                if self.verbose:
                    self.environment.render(self.agent)
                self.current_timestep += 1
                episode_timestep += 1
                current_return += reward
                if len(self.agent.memory.items) >= int(2* self.agent.memory.batch_size):
                    self.agent.learn()
            if (current_return/episode_timestep) > best_average_return:
                best_average_return = current_return
                self.agent.save_models(f"{self.environment.output_path}/models")
            if is_done:
                reward_sum += reward 
                # self.agent.memory.reset()
        if self.verbose:
            self.environment.create_video('world')
        return
