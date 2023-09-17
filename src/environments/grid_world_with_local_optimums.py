from typing import List, Optional, Tuple, Union, Any
from gym.core import Env, RenderFrame
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
from agent import SARSAAgent, Action
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import shutil
from .grid_world import GridWorld

class LocalUptimumGridWorld(GridWorld):

    def reset(self):
        np.random.seed(self.seed)
        self.world: np.ndarray = 1 - 2 * np.random.uniform(size=self.world.shape)
        self.world[self.world == self.world.max()] = 2  # terminal states
        self.world[abs(self.world) < 0.1] = 0
        self.world[self.world > 0] = 1 # local optimum
        self.world[self.world == self.world.min()] = -2  # non-terminal states
        self.world[self.world < 0] = -1 # local bad state
        self.world_best = np.where(self.world == 2)
        self.world_worst = np.where(self.world == -2)
        self.world_best = np.array([self.world_best[0][0], self.world_best[1][0]])
        self.world_worst = np.array([self.world_worst[0][0], self.world_worst[1][0]])
        self.wind = (np.random.uniform(self.wind.shape) > 0.5).astype(int)
        return self.agent_start_position.copy()  # x_0 = 0 , y_0 = 0

    def render_world(self, agent: SARSAAgent) -> None:
        x, y = int(agent.position[0]), int(agent.position[1])
        world_copy = self.world.copy()
        world_copy[x, y] = 3
        q_world = agent.q.copy()
        for i in range(len(q_world)):
            for j in range(len(q_world[0])):
                quailities = list(np.round(q_world[i][j], decimals=1).astype(str))
                plt.text(i - 0.5, j, f"{' '.join(quailities)}", fontdict={"size": 5})
        colors = [
            "#f50505",  # red -2
            "#f54d05",  # orange -1
            'f7ec11' ,  # yellow 0
            "#34ebb1",  # light green 1
            "#34eb52",  # green 2 
            "#05b1f5",  # blue (agent) 3
        ]
        cmap = plt.colors.ListedColormap(colors)
        plt.imshow(world_copy, cmap=cmap)
        plt.grid(True, color="black", linewidth=0.5)
        # Set ticks and labels
        plt.xticks(np.arange(0.5, self.world.shape[0], 1), range(self.world.shape[0]))
        plt.yticks(np.arange(0.5, self.world.shape[1], 1), range(self.world.shape[1]))
        plt.xlabel("Column")
        plt.ylabel("Row")
        np_array = self.get_plot_array()
        Image.fromarray(np_array).save(
            f"{self.output_path}/imgs/{self.current_timestep}_world.png"
        )