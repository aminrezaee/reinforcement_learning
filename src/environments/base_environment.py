import os
from typing import Any, Tuple

import numpy as np
from gym.core import Env
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PIL import Image

from action import Action
from agents.agent import Agent


class BaseEnvironment(Env):
    def __init__(self) -> None:
        pass

    def is_done(self, x:int, y:int) -> bool:
        return NotImplementedError
    
    def step(self, agent:Agent , maximum_timesteps) -> Tuple[Any, float, bool, bool, dict]:
        return NotImplementedError
    
    def reset(self) -> None:
        return NotImplementedError
    
    def invalid_move(self , action:Action , agent:Agent) -> bool:
        return NotImplementedError
    
    def render(self , agent:Agent , agent_color:int) -> None:
        return NotImplementedError
    
    def get_plot_array(self):
        canvas = self.ax.figure.canvas.manager.canvas
        # Update the canvas to render the plot
        canvas.draw()
        # Convert the plot to a 2D NumPy array
        plot_array = np.array(canvas.renderer.buffer_rgba())
        self.ax.cla()
        return plot_array
    
    def create_video(self , postfix:str):
        files = os.listdir(f"{self.output_path}/imgs")
        files = [file_name for file_name in files if postfix in file_name]
        files = sorted(files, key=lambda x: int(x.split(".")[0].split("_")[0]))
        images = [np.array(Image.open(f"{self.output_path}/imgs/{name}")) for name in files]
        clip = ImageSequenceClip(images, fps=8)
        clip.write_videofile(f"{self.output_path}/output_{postfix}.mp4")
        return
    
    def hex_to_rgba(self, hex_color:str) -> Tuple:
        # Remove the "#" symbol if present
        hex_color = hex_color.lstrip("#")
        
        # Convert the hexadecimal color code to decimal values
        red = int(hex_color[0:2], 16)
        green = int(hex_color[2:4], 16)
        blue = int(hex_color[4:6], 16)
        
        # Convert the decimal values to the RGBA format
        rgba_color = np.array([red, green, blue, 255])
    
        return rgba_color/255

