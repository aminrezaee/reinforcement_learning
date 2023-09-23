import numpy as np

from .agent import Agent


class QLearningAgent(Agent):

    def step(self, reward:float , new_position:np.ndarray , current_timestep:int) -> None:
        x_0 , x_1 , y_0 , y_1 , _ = self._act(new_position , current_timestep)
        self.q[ y_0 , x_0 , self.action.value] += self.alpha * ( # q(s , a) = q(s , a) + alpha * ( reward + gamma * (q(s' , a') - q(s , a))
            reward + self.discount_rate * (self.q[y_1 , x_1].max()) - self.q[y_0 , x_0 , self.action.value])
        new_action = self.act(current_timestep)
        self.action = new_action
        return    