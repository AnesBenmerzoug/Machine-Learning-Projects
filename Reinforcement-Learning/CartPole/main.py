import faulthandler
from pathlib import Path
import time

import click

from src import AgentTrainer, AgentTester, plot_box


class Parameters:
    # Model Parameters
    model_dir = Path("./trained_models/").absolute()
    # Dataset Parameters
    num_workers = 2
    # Training Parameters
    max_num_iterations = 1000
    max_num_timeteps = 100000
    target_episode_reward_mean = 100
    timesteps_per_iteration = 500
    batch_size = 16
    # Optimizer Parameters
    learning_rate = 5e-4
    # Testing Parameters
    num_testing_episodes = 100


@click.command()
@click.option("--train/--no-train", is_flag=True, default=True)
@click.option("--gpu/--no-gpu", is_flag=True, default=True)
def main(train: bool, gpu: bool):
    print("Starting time: {}".format(time.asctime()))

    # To have a more verbose output in case of an exception
    faulthandler.enable()

    Parameters.train_model = train
    Parameters.use_gpu = gpu

    if Parameters.train_model is True:
        # Training the model
        trainer = AgentTrainer(Parameters)
        trainer.train_model()

    # Testing the trained model
    tester = AgentTester(Parameters)
    # Testing the policy
    rewards = tester.test_model()
    plot_box(rewards)

    print("Finishing time: {}".format(time.asctime()))


if __name__ == "__main__":
    main()
