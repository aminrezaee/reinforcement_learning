import os
import shutil
from typing import Any, Tuple

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from agents.agent import Action, Agent

from .base_environment import BaseEnvironment


class GridWorld(BaseEnvironment):
    def __init__(self , size , output_path:str , seed=0 , redo=True) -> None:
        super().__init__(output_path)
        self.seed = seed
        self.current_timestep = 0
        np.random.seed(self.seed)
        self.world:np.ndarray = np.zeros(size) # shows terminal states and rewards
        self.world_best = None
        self.world_worst = None
        self.agent_start_position = np.zeros(2)
        if redo:
            shutil.rmtree(self.output_path)
        os.makedirs(f"{self.output_path}/imgs" , exist_ok=True)
        self.fig , self.ax = plt.subplots(figsize=(2*size[0] , 2*size[1]))
        
        
    def reset(self) -> None:
        np.random.seed(self.seed)
        self.world:np.ndarray = (1 - 2* np.random.uniform(size=self.world.shape))
        self.world[self.world == self.world.max()] = 1 # terminal states
        self.world[self.world == self.world.min()] = -1 # non-terminal states
        self.world[(self.world> -1) * (self.world < 1)] = 0
        self.world_best = np.where(self.world == 1)
        self.world_best = np.array([self.world_best[0][0] , self.world_best[1][0]])
        self.world_worst = np.where(self.world == -1)
        self.world_worst = np.array([self.world_worst[0][0], self.world_worst[1][0]])
        return self.agent_start_position.copy() # x_0 = 0 , y_0 = 0
    
    def is_done(self, x:int, y:int):
        return ((x == self.world_best[0]) and (y == self.world_best[1])) or ((x == self.world_worst[0]) and (y == self.world_worst[1]))
    
    def next_state(self, action:Action , position):
        new_position = position.copy()
        if action == Action.UP:
            new_position[0] -= 1
        elif action == Action.DOWN:
            new_position[0] += 1
        elif action == Action.LEFT:
            new_position[1] -= 1
        elif action == Action.RIGHT:
            new_position[1] += 1
        else:
            raise NotImplementedError
        return new_position
    
    def step(self, agent:Agent , maximum_timesteps) -> Tuple[Any, float, bool, bool, dict]:
        action = agent.action
        if self.invalid_move(action , agent):
            reward = -5
            is_done = False
            self.current_timestep += 1
            return agent.position , reward , is_done , is_done , None
        
        new_position = self.next_state(action , agent.position)
        x , y = int(new_position[0]) , int(new_position[1])
        distance_reward = 0 if self.world_best is None else 1/(1+ np.linalg.norm(self.world_best - new_position))
        terminal_reached = self.is_done(x,y)
        coefficient = 100 if terminal_reached else 1
        reward = self.world[x, y] * coefficient - 1 + distance_reward
        is_done = terminal_reached or self.current_timestep >= maximum_timesteps
        self.current_timestep += 1
        return new_position , reward , is_done , None , None
    
    def invalid_move(self , action:Action , agent:Agent) -> bool:
        is_invalid = ((action == Action.UP) and (agent.position[0] == 0)) or \
                     ((action == Action.DOWN) and (agent.position[0] >= int(self.world.shape[0] - 1))) or \
                     ((action == Action.LEFT) and (agent.position[1] == 0)) or \
                     ((action == Action.RIGHT) and (agent.position[1] >= int(self.world.shape[1] - 1)))
        # print(agent.position , action , 'is_invalid:' + str(is_invalid))
        return is_invalid
    
    def render(self , agent:Agent , agent_color:int = 3) -> None:
        self.render_world(agent , agent_color)
        return
    

    def render_world(self,agent:Agent , agent_color:int) -> None:
        x , y = int(agent.position[0]) , int(agent.position[1])
        current_q = agent.get_q()
        print(current_q)
        agent.q[x,y] = current_q
        world_copy = self.world.copy()
        world_copy[x , y] = agent_color
        self.ax.figure(figsize=(2*world_copy.shape[0] , 2*world_copy.shape[1]))
        q_world = agent.q.copy()
        for i in range(len(q_world)):
            for j in range(len(q_world[0])):
                quailities = list(np.round(q_world[i][j],decimals=1).astype(str))
                delimiter = '   '
                texts = delimiter.join(quailities[:2]) , delimiter.join(quailities[2:])
                self.ax.text(i - 0.5, j, f"{texts[0]} \n {texts[1]}", fontdict={"size": 20} , horizontalalignment='center', verticalalignment='center')

        self.ax.imshow(world_copy, cmap='cool')
        self.ax.colorbar()
        self.ax.grid(True, color='black', linewidth=0.5)
        # Set ticks and labels
        self.ax.xticks(np.arange(0.5, self.world.shape[0], 1), range(self.world.shape[0]))
        self.ax.yticks(np.arange(0.5, self.world.shape[1], 1), range(self.world.shape[1]))
        self.ax.xlabel('Column')
        self.ax.ylabel('Row')
        np_array = self.get_plot_array()
        Image.fromarray(np_array).save(f"{self.output_path}/imgs/{self.current_timestep}_world.png") 
