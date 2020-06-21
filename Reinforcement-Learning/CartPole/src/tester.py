import time
import logging
from typing import TYPE_CHECKING

from .agent import AgentNetwork
from .environment import cartpole_pixel_env_creator

import ray
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.registry import register_env
import torch
from loguru import logger


if TYPE_CHECKING:
    from typing import Tuple, List

custom_env_name = "CartPole-Pixel"

register_env(custom_env_name, cartpole_pixel_env_creator)


class AgentTester:
    def __init__(self, parameters):
        self.params = parameters

        # Checking for GPU
        self.use_gpu = self.params.use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")

        # Register model
        ModelCatalog.register_custom_model("agent_network", AgentNetwork)

        # Get environment
        self.env = cartpole_pixel_env_creator({"shape": (40, 60, 3), "num_stack": 4},)

        self.config = {
            "model": {
                "custom_model": "agent_network",
                "custom_options": {"shape": (40, 60, 3), "num_stack": 4},
            },
            "env": custom_env_name,
            "env_config": {"shape": (40, 60, 3), "num_stack": 4},
            "num_workers": self.params.num_workers,
            "num_gpus": self.use_gpu,
            "use_pytorch": True,
            "log_level": logging.INFO,
        }

    def test_model(self) -> Tuple[List[float], list]:
        ray.init(logging_level=logging.INFO, ignore_reinit_error=True)
        agent = DQNTrainer(self.config, env=custom_env_name)
        weights = torch.load(
            self.params.model_dir / "trained_model.pt",
            map_location=lambda storage, loc: storage,
        )
        agent.set_weights({"default_policy": weights})
        rewards = []
        longest_screens = []
        for i in range(self.params.num_testing_episodes):
            screens = []
            try:
                logger.info("Iteration: {}", i)
                state = self.env.reset()
                done = False
                cumulative_reward = 0
                while not done:
                    action = agent.compute_action(state)
                    state, reward, done, _ = self.env.step(action)
                    screen = self.env.render(mode="rgb_array")
                    screens.append(screen)
                    cumulative_reward += reward
                    time.sleep(0.01)
                logger.info("Iteration: {}, Reward: {}", i, cumulative_reward)
                rewards.append(cumulative_reward)
            except KeyboardInterrupt:
                logger.info("Testing was interrupted")
                break
            if len(screens) > len(longest_screens):
                longest_screens = screens
        self.env.close()
        ray.shutdown()
        return rewards, longest_screens
