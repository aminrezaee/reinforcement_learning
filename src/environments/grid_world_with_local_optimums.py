import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from agent import Agent
from .grid_world import GridWorld
from matplotlib.colors import ListedColormap , NoNorm

class LocalUptimumGridWorld(GridWorld):

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
        return self.agent_start_position.copy()  # x_0 = 0 , y_0 = 0

    def render_world(self, agent: Agent) -> None:
        x, y = int(agent.position[0]), int(agent.position[1])
        world_copy = self.world.copy()
        world_copy[x, y] = 3
        q_world = agent.q.copy()
        plt.figure(figsize=(2*world_copy.shape[0] , 2*world_copy.shape[1]))
        for i in range(len(q_world)):
            for j in range(len(q_world[0])):
                quailities = list(np.round(q_world[i][j], decimals=1).astype(str))
                delimiter = '   '
                texts = delimiter.join(quailities[:2]) , delimiter.join(quailities[2:])
                plt.text(i - 0.5, j, f"{texts[0]} \n {texts[1]}", fontdict={"size": 20} , horizontalalignment='center', verticalalignment='center')
        colors = [
            "#f50505",  # red -2
            "#f54d05",  # orange -1
            'f7ec11' ,  # yellow 0
            "#34ebb1",  # light green 1
            "#34eb52",  # green 2 
            "#05b1f5",  # blue (agent) 3
        ]
        rgba_colors = [self.hex_to_rgba(color) for color in colors]
        cmap = ListedColormap(rgba_colors)
        plt.grid(True, color="black", linewidth=0.5)
        # Set ticks and labels
        plt.xticks(np.arange(0.5, self.world.shape[0], 1), range(self.world.shape[0]))
        plt.yticks(np.arange(0.5, self.world.shape[1], 1), range(self.world.shape[1]))
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.imshow((world_copy -world_copy.min()).astype(np.uint8), cmap=cmap , norm=NoNorm())
        np_array = self.get_plot_array()
        Image.fromarray(np_array).save(
            f"{self.output_path}/imgs/{self.current_timestep}_world.png"
        )