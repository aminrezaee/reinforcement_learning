import numpy as np

from .agent import Agent


class SARSAAgent(Agent):
    
    def step(self, reward:int , new_position:np.ndarray , current_timestep:int) -> None:
        x_0 , x_1 , y_0 , y_1 , new_action = self._act(new_position , current_timestep)
        self.q[ y_0 , x_0 , self.action.value] += self.alpha * ( # q(s , a) = q(s , a) + alpha * ( reward + gamma * (q(s' , a') - q(s , a))
            reward + self.discount_rate * (self.q[y_1 , x_1 , new_action.value]) - self.q[y_0 , x_0 , self.action.value])
        self.action = new_action

    def learn(self) -> None:
        return 

class ExpectedSARSA(Agent):
    def step(self, reward: int, new_position: np.ndarray, current_timestep: int) -> None:
        x_0 , x_1 , y_0 , y_1 , _ = self._act(new_position , current_timestep , choose_next_action=False)
        values = self.q[y_1 , x_1]
        action_count = len(values)
        max_value = values.max()
        greedy_action_count = sum(self.q[y_1, x_1] == max_value)
        if greedy_action_count == action_count:
            actions_probabilities = [1/action_count for _ in values]
        else:
            greedy_prob = (1 - self.epsilon * ((action_count-1)/(action_count)))/greedy_action_count
            random_prob = self.epsilon/action_count
            actions_probabilities = [greedy_prob if value == max_value else random_prob for value in values]
        mean_q_value  = (np.array(actions_probabilities) * self.q[y_1 , x_1]).sum()
        self.q[ y_0 , x_0 , self.action.value] += self.alpha * ( # q(s , a) = q(s , a) + alpha * ( reward + gamma * (q(s' , a') - q(s , a))
            reward + self.discount_rate * mean_q_value - self.q[y_0 , x_0 , self.action.value])
        self.action = self.act(current_timestep)
        return 
    
    def learn(self) -> None:
        return 
