from typing import TYPE_CHECKING

from loguru import logger
from ray.rllib.agents.callbacks import DefaultCallbacks
import torch

if TYPE_CHECKING:
    from typing import Union
    from pathlib import Path
    from ray.rllib.agents.trainer import Trainer


class TorchModelStoreCallbacks(DefaultCallbacks):
    def __init__(self, model_path: "Union[str, Path]"):
        super(TorchModelStoreCallbacks, self).__init__()
        self.model_path = model_path

    def on_train_result(self, trainer: Trainer, result: dict, **kwargs):
        iteration = result["training_iteration"]
        logger.info(f"Iteration {iteration}")
        if iteration % 10 == 0:
            logger.info(f"Model checkpoint at iteration {iteration}")
            torch.save(trainer.get_weights()["default_policy"], self.model_path)
