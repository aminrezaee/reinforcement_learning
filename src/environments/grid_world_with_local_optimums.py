from typing import Any, Tuple

import numpy as np
from matplotlib.colors import ListedColormap, NoNorm
from PIL import Image

from action import Action
from agents.agent import Agent

from .grid_world import GridWorld


class LocalUptimumGridWorld(GridWorld):
    def __init__(self, size, output_path: str, seed=0, redo=True) -> None:
        super().__init__(size, output_path, seed, redo)
        self.current_timestep_in_episode = 0

    def reset(self):
        np.random.seed(self.seed)
        self.world: np.ndarray = 1 - 2 * np.random.uniform(size=self.world.shape)
        self.world[self.world == self.world.max()] = 2  # terminal states
        self.world[abs(self.world) < 0.1] = 0
        self.world[(self.world > 0) * (self.world < 1.0)] = 1 # local optimum
        self.world[self.world == self.world.min()] = -2  # non-terminal states
        self.world[(self.world < 0) * (self.world > -1.0)] = -1 # local bad state
        self.world_best = np.where(self.world == 2)
        self.world_worst = np.where(self.world == -2)
        self.world_best = np.array([self.world_best[0][0], self.world_best[1][0]])
        self.world_worst = np.array([self.world_worst[0][0], self.world_worst[1][0]])
        self.current_timestep_in_episode = 0
        return self.agent_start_position.copy()  # x_0 = 0 , y_0 = 0
    
    def step(self, agent:Agent , maximum_timesteps) -> Tuple[Any, float, bool, bool, dict]:
        action = agent.action
        if self.invalid_move(action , agent):
            reward = -5 #- min(1.2 ** self.current_timestep_in_episode , 100)
            is_done = False
            self.current_timestep += 1
            return agent.position , reward , is_done , is_done , None
        new_position = self.next_state(action , agent.position)
        x , y = int(new_position[0]) , int(new_position[1])
        distance_reward = 0 if self.world_best is None else 1/(1+ np.linalg.norm(self.world_best - np.argmax(new_position)))
        terminal_reached = self.is_done(x,y)
        coefficient = 100 if terminal_reached else 1
        reward = self.world[x, y] * coefficient - 1 #+ distance_reward
        is_done = terminal_reached or self.current_timestep >= maximum_timesteps
        self.current_timestep += 1
        self.current_timestep_in_episode += 1
        return new_position , reward , is_done , None , None
    
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

    def render_world(self, agent: Agent , agent_color:int) -> None:
        x, y = int(agent.position[0]), int(agent.position[1])
        world_copy = self.world.copy()
        world_copy[x, y] = agent_color
        q_world = agent.q.copy()
        for i in range(len(q_world)):
            for j in range(len(q_world[0])):
                qualities = list(q_world[i][j].astype(np.int64).astype(str))
                delimiter = '   '
                texts = delimiter.join(qualities[:2]) , delimiter.join(qualities[2:])
                self.ax.text(i - 0.5, j, f"{texts[0]} \n{texts[1]}", fontdict={"size": 20} , verticalalignment='center')
        colors = [
            "#f50505",  # red -2
            "#f54d05",  # orange -1
            'f7ec11' ,  # yellow 0
            "#34ebb1",  # light green 1
            "#34eb52",  # green 2 
            "#05b1f5",  # blue (agent) 3
            "#8cd7f5" , # light blue (simulated agent) 4
        ]
        rgba_colors = [self.hex_to_rgba(color) for color in colors]
        cmap = ListedColormap(rgba_colors)
        self.ax.grid(True, color="black", linewidth=0.5)
        # Set ticks and labels
        self.ax.set_xticks(np.arange(0.5, self.world.shape[0], 1), range(self.world.shape[0]))
        self.ax.set_yticks(np.arange(0.5, self.world.shape[1], 1), range(self.world.shape[1]))
        self.ax.set_xlabel("Column")
        self.ax.set_ylabel("Row")
        self.ax.imshow((world_copy -world_copy.min()).astype(np.uint8), cmap=cmap , norm=NoNorm())
        np_array = self.get_plot_array()
        Image.fromarray(np_array).save(
            f"{self.output_path}/imgs/{self.current_timestep}_world.png"
        )