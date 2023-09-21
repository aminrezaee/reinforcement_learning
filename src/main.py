from environments.grid_world_with_local_optimums import LocalUptimumGridWorld as GridWorld
from agent import SARSAAgent , ExpectedSARSA , QLearningAgent , Agent , DynaQAgent
from argparse import ArgumentParser , Namespace
from tqdm import tqdm
from logging import getLogger
import logging

logger = getLogger()
logger.setLevel(logging.INFO)

def main():
    environment = GridWorld((5,5) , output_path='./../outputs')
    agent = DynaQAgent(environment.agent_start_position , 
                                environment.world.shape,
                                epsilon= 0.15,
                                alpha= 0.5 ,
                                discount_rate=1)
    parser = ArgumentParser()
    parser.add_argument('-t' , '--timesteps' , default=2000 , type=int)
    args = parser.parse_args()
    model_based(environment , agent , args)
    
    
    return

def model_based(environment:GridWorld , agent:DynaQAgent , args:Namespace):
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
                agent.model.update(agent.state_optimizer , agent.reward_optimizer) # update model to be more exact
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
    main()