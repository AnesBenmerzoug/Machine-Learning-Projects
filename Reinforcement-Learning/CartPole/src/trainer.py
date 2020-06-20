from functools import partial
import logging
from loguru import logger

from .agent import AgentNetwork
from .environment import cartpole_pixel_env_creator
from .callback import TorchModelStoreCallbacks

import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.registry import register_env
import torch

custom_env_name = "CartPole-Pixel"

# Register the custom environment
register_env(custom_env_name, cartpole_pixel_env_creator)


class AgentTrainer:
    def __init__(self, parameters):
        self.params = parameters

        # Checking for GPU
        self.use_gpu = self.params.use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")

        # Register model
        ModelCatalog.register_custom_model("agent_network", AgentNetwork)

        # Training configuration
        self.config = {
            "model": {
                "custom_model": "agent_network",
                "custom_options": {"shape": (40, 60, 3), "num_stack": 4},
            },
            "env": custom_env_name,
            "env_config": {"shape": (40, 60, 3), "num_stack": 4},
            "callbacks": partial(
                TorchModelStoreCallbacks,
                model_path=self.params.model_dir / "trained_model.pt",
            ),
            "double_q": True,
            "dueling": True,
            "noisy": True,
            "n_step": 10,
            "lr": self.params.learning_rate,
            "train_batch_size": self.params.batch_size,
            "buffer_size": self.params.replay_buffer_size,
            "prioritized_replay": True,
            "num_workers": self.params.num_workers,
            "num_gpus": self.use_gpu,
            "use_pytorch": True,
            "log_level": logging.INFO,
        }

        # Stopping conditions
        self.stop_conditions = {
            "training_iteration": self.params.max_num_iterations,
            "timesteps_total": self.params.max_num_timeteps,
            "episode_reward_min": self.params.target_episode_reward,
        }

    def train_model(self):
        ray.init(logging_level=logging.INFO)
        logger.info("Starting training")
        trials = tune.run(
            DQNTrainer,
            config=self.config,
            stop=self.stop_conditions,
            verbose=1,
            return_trials=True,
        )
        logger.info("Finished training")
        return trials
