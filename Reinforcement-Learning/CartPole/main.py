import faulthandler
from pathlib import Path
import time

import click

from src import *


class Parameters:
    # Model Parameters
    model_dir = Path("./trained_models/")
    # Replay Memory Parameters
    replay_memory_capacity = 10000
    # Epsilon Greedy Policy Parameters
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 500
    # Dataset Parameters
    dataset_dir = Path("./data/")
    num_workers = 2
    # Training Parameters
    num_episodes = 1000
    num_steps_per_episode = 1000
    gamma = 0.999
    target_update = 50
    batch_size = 64
    # Optimizer Parameters
    optimizer = "Adam"
    learning_rate = 1.0e-3
    momentum = 0.9
    nesterov = True
    # Scheduler Parameters
    decay_coeff = 0.5
    step_size = 1000


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
        # Instantiating the trainer
        trainer = AgentTrainer(Parameters)
        # Training the model
        avg_losses, episode_durations = trainer.train_model()
        # Plot losses
        plotlosses(
            avg_losses,
            title="Average Loss per Episode",
            xlabel="Episode",
            ylabel="Average Loss",
        )
        # Plot durations
        plotlosses(
            episode_durations,
            title="Episode Durations",
            xlabel="Episode",
            ylabel="Duration",
        )

    # Instantiating the test
    tester = AgentTester(Parameters)
    # Testing the policy
    tester.test_model()

    print("Finishing time: {}".format(time.asctime()))


if __name__ == "__main__":
    main()
