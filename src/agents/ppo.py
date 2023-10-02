import logging
from typing import Tuple , List

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.optim import Adam

from action import Action
from models.ppo.actor import Actor
from models.ppo.critic import Critic
from replay_memory import MemoryItem

from .agent import Agent


class ProximalPolicyOptimization(Agent):
    def __init__(
        self,
        start_position: np.ndarray,
        world_map_size: tuple,
        batch_size:int,
        epsilon=0.05,
        alpha=0.1,
        discount_rate=0.8,
        iterations_per_update: int = 3,
        gae_lambda:float = 0.95,
        clip_thr:float = 0.2 ,
        epochs:int=3,
        device:str = 'cpu'
    ) -> None:
        super().__init__(start_position, world_map_size , batch_size , epsilon, alpha, discount_rate , device)
        self.actor = Actor(len(start_position), len(Action.get_all_actions()) , device)
        self.critic = Critic(len(start_position), len(Action.get_all_actions()) , device)
        self.actor_optimizer = Adam(params=self.actor.parameters() , lr=1e-6)
        self.critic_optimizer = Adam(params=self.critic.parameters() , lr=1e-5)
        self.iterations_per_update = iterations_per_update
        self.gae_lambda = gae_lambda
        self.clip_thr = clip_thr
        self.epochs = epochs

    def step(self, new_position: np.ndarray) -> Tuple[Action, float , float]:
        self.position = new_position
        state = Tensor(new_position)[None,:]
        self.actor.eval()
        self.critic.eval()
        distribution: Categorical = self.actor(state)
        value = self.critic(state)
        action = distribution.sample()
        self.action = Action.get_all_actions()[action]
        log_prob = torch.squeeze(distribution.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        return self.action , log_prob , value

    def learn(self) -> None:
        items:List[MemoryItem] = list(self.memory.items)
        advantages =  np.zeros(len(items) , dtype=np.float32)
        values = Tensor([items[i].value for i in range(len(items))])
        rewards = Tensor([items[i].reward for i in range(len(items))])
        dones = Tensor([items[i].done for i in range(len(items))])
        for t in range(int(len(rewards)-1)):
            discount = 1
            a_t = 0
            for k in range(t , int(len(rewards) - 1)):
                a_t += discount * (rewards[k] + self.discount_rate * values[int(k+1) *(1-int(dones[k]))]) - values[k]
                discount *= (self.discount_rate * self.gae_lambda)
            advantages[t] = a_t
        advantages = Tensor(advantages , device= self.device)
        for _ in range(self.epochs):
            for iter in range(self.iterations_per_update):
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                indices , states, actions, log_probs, values, rewards, dones = self.memory.sample(self.device)
                distribution:Categorical = self.actor(states)
                critic_values = torch.squeeze(self.critic(states))
                new_log_probs = distribution.log_prob(actions)
                prob_ratio = new_log_probs.exp()/log_probs.exp()
                weighted_probs = advantages[indices] * prob_ratio
                clipped_probs = torch.clamp(prob_ratio , 1-self.clip_thr , 1+ self.clip_thr) * advantages[indices]
                actor_loss = -torch.min(weighted_probs , clipped_probs).mean()
                returns = advantages[indices] + values
                critic_loss = (returns - critic_values) ** 2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5 * critic_loss
                total_loss.backward()
                loss_text = f"loss:{round(total_loss.item() , ndigits=3)}"
                logging.getLogger().log(logging.INFO, loss_text)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.memory.reset()
        return
    
    def get_q(self , positions:np.ndarray) -> np.ndarray:
        self.actor.eval()
        return self.actor(Tensor(positions ,device=self.device)).probs.detach().cpu().numpy() * 10
    
    def save_models(self , path) -> None:
        torch.save(self.actor.state_dict(),f"{path}/actor.pt")
        torch.save(self.critic.state_dict(),f"{path}/critic.pt")
        return 
