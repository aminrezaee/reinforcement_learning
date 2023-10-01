from argparse import ArgumentParser

import numpy as np

from agents.ppo import ProximalPolicyOptimization
from environments.grid_world_with_local_optimums import \
    LocalUptimumGridWorld as GridWorld
from trainer.base_trainer import BaseTrainer


def main():
    parser = ArgumentParser()
    parser.add_argument("--timesteps", "-t", default=2000, type=int)
    args = parser.parse_args()
    environment = GridWorld((10, 10), output_path="./../outputs/")
    agent = ProximalPolicyOptimization(
        start_position=np.array([0, 0]),
        world_map_size=environment.world.shape,
        batch_size=20,
    )
    trainer = BaseTrainer(agent, environment, args.timesteps)
    trainer.train()
    return


if __name__ == "__main__":
    main()
