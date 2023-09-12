from environments.windy_grid_world import WindyGridWorld
from agent import SARSAAgent
from argparse import ArgumentParser
def main():
    environment = WindyGridWorld((8,16) , output_path='./../outputs')
    agent = SARSAAgent(environment.agent_start_position , 
                                environment.world.shape,
                                epsilon= 0.05,
                                alpha= 0.1 ,
                                discount_rate=1)
    parser = ArgumentParser()
    parser.add_argument('-t' , '--timesteps' , default=400 , type=int)
    args = parser.parse_args()
    while environment.current_timestep < args.timesteps:
        is_done = False
        first_position = environment.reset()
        agent.position = first_position
        agent.action = agent.act() # returns a new action
        while not is_done:
            new_position , reward , is_done , _ , _ = environment.step(agent)
            agent.step(reward , new_position) # updates values and creates new action
            environment.render(agent)
    environment.create_video()
    return

if __name__ == "__main__":
    main()