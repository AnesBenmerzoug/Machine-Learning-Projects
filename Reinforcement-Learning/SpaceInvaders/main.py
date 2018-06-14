from __future__ import print_function, division
from collections import namedtuple
import faulthandler
import yaml
import time

from src import *

if __name__ == "__main__":
    print("Starting time: {}".format(time.asctime()))

    # To have a more verbose output in case of an exception
    faulthandler.enable()

    with open('parameters.yaml', 'r') as params_file:
        parameters = yaml.safe_load(params_file)
        parameters = namedtuple('Parameters', (parameters.keys()))(*parameters.values())

    if parameters.trainModel is True:
        # Instantiating the trainer
        trainer = AgentTrainer(parameters)
        # Training the model
        avg_losses, episode_durations = trainer.train_model()
        # Plot losses
        plotlosses(avg_losses, title='Average Loss per Episode', xlabel='Episode', ylabel='Average Loss')
        # Plot durations
        plotlosses(episode_durations, title='Episode Durations', xlabel='Episode', ylabel='Duration')

    else:
        # Instantiating the test
        tester = AgentTester(parameters)
        # Testing the policy
        tester.test_model()

    print("Finishing time: {}".format(time.asctime()))

