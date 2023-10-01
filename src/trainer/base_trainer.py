from agents.agent import Agent
from action import Action
from environments.base_environment import BaseEnvironment
import logging
class BaseTrainer:
    def __init__(self , 
                 agent:Agent , 
                 environment:BaseEnvironment, 
                 maximum_timesteps:int , 
                 verbose:bool = True , 
                 update_per_timestep:int = 10) -> None:
        self.agent = agent 
        self.environment = environment
        self.maximum_timesteps = maximum_timesteps
        self.current_timestep:int = 0
        self.update_per_timestep = update_per_timestep
        self.verbose = verbose
    
    def train(self):
        reward = 0
        reward_sum = 0
        while self.current_timestep < self.maximum_timesteps:
            is_done = False
            position = self.environment.reset()
            while not is_done:
                logging.getLogger().log(logging.DEBUG ,f"timestep:{self.current_timestep}")
                action , prob , value = self.agent.step(position)
                new_position , reward , is_done , _ , _ = self.environment.step(self.agent , self.maximum_timesteps)
                self.agent.memory.append(new_position , prob , value , action , reward , is_done)
                if self.verbose:
                    self.environment.render(self.agent)
                if self.current_timestep % self.update_per_timestep == 0:
                    self.agent.learn()
                self.current_timestep += 1
            if is_done:
                reward_sum += reward 
            self.agent.memory.reset()
        if self.verbose:
            self.environment.create_video('world')
        return
