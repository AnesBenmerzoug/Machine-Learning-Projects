import torch
from .agent import AgentNetwork
from .environment import cartpole_pixel_env_creator
import logging
from loguru import logger

import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.registry import register_env

custom_env_name = "CartPole-Pixel"

register_env(custom_env_name, cartpole_pixel_env_creator)


class AgentTrainer:
    def __init__(self, parameters):
        self.params = parameters

        # Checking for GPU
        self.use_gpu = self.params.use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")

        # Register model
        ModelCatalog.register_custom_model("agent_network", AgentNetwork)

        self.config = {
            "model": {
                "custom_model": "agent_network",
                "custom_options": {"shape": (40, 60, 3), "dueling": True,},
            },
            "env": custom_env_name,
            "env_config": {"shape": (40, 60, 3),},
            "lr": self.params.learning_rate,
            "train_batch_size": self.params.batch_size,
            "num_workers": self.params.num_workers,
            "num_gpus": self.use_gpu,
            "use_pytorch": True,
            "log_level": logging.INFO,
        }

        self.stop_conditions = {
            "training_iteration": self.params.max_num_iterations,
            "timesteps_total": self.params.max_num_timeteps,
            "episode_reward_mean": self.params.target_episode_reward_mean,
        }

    def train_model(self):
        ray.init(logging_level=logging.INFO)
        logger.info("Starting training")
        tune.run(
            self._train_function,
            config=self.config,
            stop=self.stop_conditions,
            verbose=1,
        )
        logger.info("Finished training")

    def _train_function(self, config: dict, reporter):
        trainer = DQNTrainer(env=custom_env_name, config=config)
        logger.info(f"model summary: \n {trainer.get_policy().model}")
        for i in range(1, self.params.max_num_iterations):
            try:
                logger.info(f"Iteration {i}")
                result = trainer.train()
                reporter(**result)
                if i % 10 == 0:
                    logger.info(f"Model checkpoint at iteration {i}")
                    model = trainer.get_policy().model
                    torch.save(
                        model.state_dict(), self.params.model_dir / "trained_model.pt",
                    )
            except KeyboardInterrupt:
                print("Training was interrupted")
                break
        logger.info("Saving model at end of worker training")
        model = trainer.get_policy().model
        torch.save(
            model.state_dict(), self.params.model_dir / "trained_model.pt",
        )
        trainer.stop()
