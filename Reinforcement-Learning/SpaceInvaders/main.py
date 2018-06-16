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
        plot_losses(avg_losses, title='Average Loss per Episode', xlabel='Episode', ylabel='Average Loss')
        # Plot durations
        plot_losses(episode_durations, title='Episode Durations', xlabel='Episode', ylabel='Duration')

    else:
        # Instantiating the test
        tester = AgentTester(parameters)
        # Testing the policy
        screens, scores = tester.test_model()
        # Plot Scores
        plot_scores(scores, xlabel='Score', ylabel='Number of Games', bins=8)
        # Save animation
        save_animation('static', screens, 10)

    print("Finishing time: {}".format(time.asctime()))

