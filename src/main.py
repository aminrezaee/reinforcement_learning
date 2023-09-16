from environments.windy_grid_world import WindyGridWorld
from agent import SARSAAgent
from argparse import ArgumentParser
from tqdm import tqdm
def main():
    environment = WindyGridWorld((5,5) , output_path='./../outputs')
    agent = SARSAAgent(environment.agent_start_position , 
                                environment.world.shape,
                                epsilon= 0.15,
                                alpha= 0.3 ,
                                discount_rate=1)
    parser = ArgumentParser()
    parser.add_argument('-t' , '--timesteps' , default=300 , type=int)
    args = parser.parse_args()
    reward_sum = 0
    reward = 0
    while environment.current_timestep < args.timesteps:
        is_done = False
        print(f"resetting:{reward_sum}")
        first_position = environment.reset()
        agent.position = first_position
        # print(agent.position)
        environment.render(agent)
        agent.action = agent.act(environment.current_timestep) # returns a new action a_0
        while not is_done:
            if environment.current_timestep == 224:
                print("a")
            new_position , reward , is_done , _ , _ = environment.step(agent , args.timesteps) # r_0
            print(environment.current_timestep)
            agent.step(reward , new_position , environment.current_timestep) # updates values and creates new action s_1 , a_1
            if is_done:
                reward_sum += reward 
            environment.render(agent)
    environment.create_video('world')
    return

if __name__ == "__main__":
    main()