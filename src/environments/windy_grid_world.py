from typing import List, Optional, Tuple, Union , Any
from gym.core import Env,RenderFrame
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image 
import os
from agent import SARSAAgent , Action
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

class WindyGridWorld(Env):
    def __init__(self , size , output_path:str , seed=0) -> None:
        super().__init__()
        self.seed = seed
        self.output_path = output_path
        self.current_timestep = 0
        np.random.seed(self.seed)
        self.world:np.ndarray = np.zeros(size) # shows terminal states and rewards
        self.wind:np.ndarray = np.zeros(tuple(list(size) + [4])) # shows wind action
        self.agent_start_position = np.zeros(2)
        os.makedirs(f"{self.output_path}/imgs" , exist_ok=True)
        
        
    def reset(self):
        np.random.seed(self.seed)
        self.world:np.ndarray = (1 - 2* np.random.uniform(size=self.world.shape))
        self.world[self.world == self.world.max()] = 1 # terminal states
        self.world[self.world == self.world.min()] = -1 # non-terminal states
        self.world[(self.world> -1) * (self.world < 1)] = 0
        self.wind = (np.random.uniform(self.wind.shape) > 0.5).astype(int)
        return self.agent_start_position # x_0 = 0 , y_0 = 0
    
    def step(self, agent:SARSAAgent) -> Tuple[Any, float, bool, bool, dict]:
        action = agent.action
        if self.invalid_move(action , agent):
            reward = -1
            is_done = False
            return agent.position , reward , is_done , is_done , None
        x , y = int(agent.position[0]) , int(agent.position[1])
        is_done = self.world[x, y] != 0
        reward = self.world[x, y] * 100 
        new_position = agent.position
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
        self.current_timestep += 1
        return new_position , reward , is_done , None , None
    
    def invalid_move(self , action:Action , agent:SARSAAgent):
        is_invalid = ((action == Action.UP) and (agent.position[0] == 0)) or \
                     ((action == Action.DOWN) and (agent.position[0] >= int(self.world.shape[0] - 1))) or \
                     ((action == Action.LEFT) and (agent.position[1] == 0)) or \
                     ((action == Action.RIGHT) and (agent.position[1] >= int(self.world.shape[1] - 1)))
        # print(agent.position , action , 'is_invalid:' + str(is_invalid))
        return is_invalid
    
    def render(self , agent:SARSAAgent) -> None:
        x , y = int(agent.position[0]) , int(agent.position[1])
        world_copy = self.world.copy()
        world_copy[x , y] = 3
        plt.imshow(world_copy, cmap='cool')
        plt.grid(True, color='black', linewidth=0.5)
        # Set ticks and labels
        plt.xticks(np.arange(0.5, self.world.shape[0], 1), range(self.world.shape[0]))
        plt.yticks(np.arange(0.5, self.world.shape[1], 1), range(self.world.shape[1]))
        plt.xlabel('Column')
        plt.ylabel('Row')
        np_array = self.get_plot_array()
        Image.fromarray(np_array).save(f"{self.output_path}/imgs/{self.current_timestep}.png")
        return 
    
    def get_plot_array(self):
        canvas = plt.get_current_fig_manager().canvas
        # Update the canvas to render the plot
        canvas.draw()
        # Convert the plot to a 2D NumPy array
        plot_array = np.array(canvas.renderer.buffer_rgba())
        plt.clf()
        return plot_array
    
    def create_video(self):
        files = os.listdir(f"{self.output_path}/imgs")
        files = sorted(files, key=lambda x: int(x.split(".")[0]))
        images = [np.array(Image.open(f"{self.output_path}/imgs/{name}")) for name in files]
        clip = ImageSequenceClip(images, fps=2)
        clip.write_videofile(f"{self.output_path}/output.mp4")
        return

