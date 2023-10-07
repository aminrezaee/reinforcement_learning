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
        mean:np.ndarray, 
        max:np.ndarray,
        epsilon=0.05,
        alpha=0.1,
        discount_rate=0.8,
        iterations_per_update: int = 20,
        gae_lambda:float = 0.95,
        clip_thr:float = 0.2 ,
        epochs:int=3,
        device:str = 'cpu'
    ) -> None:
        super().__init__(start_position, world_map_size , batch_size , epsilon, alpha, discount_rate , device)
        self.actor = Actor(len(self.get_state(start_position)), len(Action.get_all_actions()) , device)
        self.critic = Critic(len(self.get_state(start_position)), len(Action.get_all_actions()) , device)
        self.actor_optimizer = Adam(params=self.actor.parameters() , lr=1e-4)
        self.critic_optimizer = Adam(params=self.critic.parameters() , lr=1e-3)
        self.iterations_per_update = iterations_per_update
        self.gae_lambda = gae_lambda
        self.clip_thr = clip_thr
        self.epochs = epochs
        self.mean = mean
        self.max = max

    def step(self, new_position: np.ndarray) -> Tuple[Action, float , float]:
        self.position = new_position
        state = Tensor(self.get_state(new_position))[None,:]
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            distribution: Categorical = self.actor(state)
            value = self.critic(state)
            action = distribution.sample()
            self.action = Action.get_all_actions()[action]
            log_prob = torch.squeeze(distribution.log_prob(action)).item()
            action = torch.squeeze(action).item()
            value = torch.squeeze(value).item()
            return self.action , log_prob , value

    def learn(self) -> None:
        for _ in range(self.epochs):
            losses = []
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            for iter in range(self.iterations_per_update):
                indices , states, actions, log_probs, values, rewards, dones , advantages = self.memory.sample(self.device)
                # states = states - states.mean(dim=0)
                # states = states/torch.std(states , dim=0)
                distribution:Categorical = self.actor(states)
                critic_values = torch.squeeze(self.critic(states))
                new_log_probs = distribution.log_prob(actions)
                prob_ratio = new_log_probs.exp()/log_probs.exp()
                weighted_probs = advantages * prob_ratio
                clipped_probs = torch.clamp(prob_ratio , 1-self.clip_thr , 1+ self.clip_thr) * advantages
                actor_loss = -torch.min(weighted_probs , clipped_probs).mean()
                returns = advantages + values
                critic_loss = (returns - critic_values) ** 2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5 * critic_loss
                # total_loss.backward()
                losses.append(total_loss)
                # loss_text = f"loss:{round(total_loss.item() , ndigits=3)}"
                # logging.getLogger().log(logging.INFO, loss_text)
            loss = sum(losses)/len(losses)
            loss.backward()
            loss_text = f"loss:{round(loss.item() , ndigits=3)}"
            logging.getLogger().log(logging.INFO, loss_text)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        self.memory.reset()
        return
    
    def get_q(self , positions:np.ndarray) -> np.ndarray:
        with torch.no_grad():
            self.actor.eval()
            if len(positions.shape) == 1:
                state = self.get_state(positions)
            else:
                state = np.array([self.get_state(position) for position in positions])
            return self.actor(Tensor(state ,device=self.device)).probs.detach().cpu().numpy() * 10
    
    def save_models(self , path) -> None:
        torch.save(self.actor.state_dict(),f"{path}/actor.pt")
        torch.save(self.critic.state_dict(),f"{path}/critic.pt")
        return 
