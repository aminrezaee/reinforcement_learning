import logging
from argparse import ArgumentParser, Namespace
from logging import getLogger

from agents.agent import Agent
from agents.dyna_q import DynaQAgent
from agents.dqn import DQN , DQNKeywords
from models.dyna_q_model import DynaQModel
from models.dqn_model import DQNModel
from environments.grid_world_with_local_optimums import \
    LocalUptimumGridWorld as GridWorld
from torch.optim import Adam

logger = getLogger()
logger.setLevel(logging.INFO)

def dynaQ():
    environment = GridWorld((10,10) , output_path='./../outputs')
    model = DynaQModel(state_size=int(environment.world.shape[0] * environment.world.shape[1]) , 
                               action_size= 4, # up , down , left , right
                               update_batch_count=1, 
                               batch_size=20)
    agent = DynaQAgent(environment.agent_start_position , 
                                environment.world.shape,
                                model,
                                epsilon= 0.15,
                                alpha= 0.5 ,
                                discount_rate=0.8)
    parser = ArgumentParser()
    parser.add_argument('-t' , '--timesteps' , default=2000 , type=int)
    args = parser.parse_args()
    reward = 0
    reward_sum = 0
    current_episode = 0
    while environment.current_timestep < args.timesteps:
        is_done = False
        logging.log(logging.DEBUG , f"resetting:{reward_sum}")
        first_position = environment.reset()
        agent.position = first_position
        # print(agent.position)
        environment.render(agent)
        agent.action = agent.act(environment.current_timestep) # returns a new action a_0
        while not is_done:
            new_position , reward , is_done , _ , _ = environment.step(agent , args.timesteps) # r_0
            agent.append_observation(agent.position , agent.action , reward + agent.discount_rate * agent.q[int(new_position[0]) , int(new_position[1])].max() , new_position) # previous state , last action and the reward
            logger.log(logging.DEBUG ,f"timestep:{environment.current_timestep}")
            agent.step(reward , new_position , environment.current_timestep) # updates values and creates new action : s_1 , a_1 ->>> direct RL
            if current_episode >= 1:
                reward_loss , next_state_loss = agent.model.update(agent.optimizers_dict) # update model to be more exact
                if next_state_loss < 0.02 and reward_loss < 1.0:
                    simulated_states , simulated_actions , simulated_rewards , simulated_new_states = agent.create_simulated_observations()
                    for simulated_state , simulated_action , simulated_reward , simulated_new_state in zip(simulated_states , 
                                                                                                        simulated_actions , 
                                                                                                        simulated_rewards , 
                                                                                                        simulated_new_states):
                        agent.position = simulated_state
                        agent.action = simulated_action
                        agent.step(simulated_reward , simulated_new_state , environment.current_timestep_in_episode)
            if is_done:
                reward_sum += reward 
                current_episode += 1
            
            environment.render(agent)
    environment.create_video('world')
    
    
    return


def dqn():
    environment = GridWorld((5,5) , output_path='./../outputs')
    discount_rate = 0.8
    model = DQNModel(state_size=int(environment.world.shape[0] * environment.world.shape[1]) , 
                               action_size= 4, # up , down , left , right
                               update_batch_count=1, 
                               batch_size=20 , 
                               gamma = discount_rate)
    optimizers_dict = {'optimizer':Adam(params= model.parameters() , lr=1e-3 , weight_decay=1e-4)}

    agent = DQN(environment.agent_start_position , 
                                environment.world.shape,
                                model,
                                optimizers_dict,
                                epsilon= 0.15,
                                alpha= 0.5 ,
                                discount_rate = discount_rate)
    parser = ArgumentParser()
    parser.add_argument('-t' , '--timesteps' , default=2000 , type=int)
    args = parser.parse_args()
    reward = 0
    reward_sum = 0
    while environment.current_timestep < args.timesteps:
        is_done = False
        logging.log(logging.DEBUG , f"resetting:{reward_sum}")
        first_position = environment.reset()
        agent.position = first_position
        # print(agent.position)
        environment.render(agent)
        agent.action = agent.act(environment.current_timestep) # returns a new action a_0
        while not is_done:
            new_position , reward , is_done , _ , _ = environment.step(agent , args.timesteps) # r_0
            agent.append_observation(agent.position , 
                                     agent.action , 
                                     reward , 
                                     new_position , 
                                     is_done) # previous state , last action and the reward
            logger.log(logging.DEBUG ,f"timestep:{environment.current_timestep}")
            agent.step(new_position , environment.current_timestep) # sets new position and creates new action
            if len(list(agent.model.data.keys())) > 5: # at least 10 different positions seen by agent
                agent.model.update(agent.optimizers_dict , keys=[DQNKeywords.ground_truth_q_values , DQNKeywords.rewards , DQNKeywords.next_states , DQNKeywords.terminal_stat])
            if is_done:
                reward_sum += reward 
            
            environment.render(agent)
    environment.create_video('world')
    
    
    return

def model_free(environment:GridWorld , agent:Agent , args):
    reward_sum = 0
    while environment.current_timestep < args.timesteps:
        is_done = False
        logging.log(logging.DEBUG , f"resetting:{reward_sum}")
        first_position = environment.reset()
        agent.position = first_position
        # print(agent.position)
        environment.render(agent)
        agent.action = agent.act(environment.current_timestep) # returns a new action a_0
        while not is_done:
            new_position , reward , is_done , _ , _ = environment.step(agent , args.timesteps) # r_0
            logger.log(logging.DEBUG ,f"timestep:{environment.current_timestep}")
            agent.step(reward , new_position , environment.current_timestep) # updates values and creates new action s_1 , a_1
            if is_done:
                reward_sum += reward 
            environment.render(agent)
    environment.create_video('world')
if __name__ == "__main__":
    dqn()