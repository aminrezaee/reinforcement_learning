from environments.grid_world_with_local_optimums import LocalUptimumGridWorld as GridWorld
from agent import SARSAAgent
from argparse import ArgumentParser
from tqdm import tqdm
from logging import getLogger
import logging

def main():
    environment = GridWorld((5,5) , output_path='./../outputs')
    agent = SARSAAgent(environment.agent_start_position , 
                                environment.world.shape,
                                epsilon= 0.15,
                                alpha= 0.3 ,
                                discount_rate=1)
    parser = ArgumentParser()
    parser.add_argument('-t' , '--timesteps' , default=300 , type=int)
    args = parser.parse_args()
    reward_sum = 0
    logger = getLogger()
    logger.setLevel(logging.ERROR)
    reward = 0
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
    return

if __name__ == "__main__":
    main()