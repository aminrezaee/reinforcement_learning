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
    parser.add_argument('-t' , '--timesteps' , default=1000 , type=int)
    args = parser.parse_args()
    for i in tqdm(range(args.timesteps)):
        is_done = False
        # print("resetting")
        first_position = environment.reset() # s_0
        agent.position = first_position
        # print(agent.position)
        environment.render(agent)
        agent.action = agent.act(environment.current_timestep) # returns a new action a_0
        while not is_done:
            new_position , reward , is_done , _ , _ = environment.step(agent , args.timesteps) # r_0
            agent.step(reward , new_position , environment.current_timestep) # updates values and creates new action s_1 , a_1
            environment.render(agent)
    environment.create_video('world')
    return

if __name__ == "__main__":
    main()