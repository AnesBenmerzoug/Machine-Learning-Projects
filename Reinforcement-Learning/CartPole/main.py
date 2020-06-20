from pathlib import Path
import time

import click
from loguru import logger

from src import AgentTrainer, AgentTester, plot_box, save_animation


class Parameters:
    # Model Parameters
    model_dir = Path("./trained_models/").absolute()
    # Dataset Parameters
    num_workers = 2
    # Training Parameters
    max_num_iterations = 500
    max_num_timeteps = 2000000
    target_episode_reward_mean = 500
    timesteps_per_iteration = 1000
    replay_buffer_size = 10000
    batch_size = 32
    # Optimizer Parameters
    learning_rate = 5e-4
    # Testing Parameters
    num_testing_episodes = 100


@click.command()
@click.option("--train/--no-train", is_flag=True, default=True)
@click.option("--gpu/--no-gpu", is_flag=True, default=True)
def main(train: bool, gpu: bool):
    logger.info(f"Starting time: {time.asctime()}")

    Parameters.train_model = train
    Parameters.use_gpu = gpu

    if Parameters.train_model is True:
        # Training the model
        trainer = AgentTrainer(Parameters)
        trials = trainer.train_model()
        results = [trial.last_result["episode_reward_mean"] for trial in trials]
        plot_box(results, title="Results")

    # Testing the trained model
    tester = AgentTester(Parameters)
    rewards, screens = tester.test_model()
    plot_box(rewards, title="Rewards")

    # Store animation of longest test run as gif
    save_animation("static", screens, fps=24)

    logger.info(f"Finishing time: {time.asctime()}")


if __name__ == "__main__":
    main()
