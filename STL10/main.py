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
        trainer = STL10Trainer(parameters)
        # Training the model
        avg_losses = trainer.train_model()
        # Plot losses
        plotlosses(avg_losses, title='Average Loss per Epoch', xlabel='Epoch', ylabel='Average Loss')

    else:
        # Instantiating the test
        tester = STL10Tester(parameters)
        # Testing the model
        tester.test_random_sample()
        # Testing the model accuracy
        total_accuracy, class_accuracy, confusion_matrix = tester.test_model()

        print("Total Accuracy = {:.2f}".format(total_accuracy))
        for i in range(len(class_accuracy)):
            print("Accuracy for class {}: {:.2f}".format(i, class_accuracy[i]))

        # Plot Per Class Accuracy
        plotaccuracy(class_accuracy, classes=tester.classes, title='Classification Accuracy per Class',
                     xlabel='Class', ylabel='Accuracy')

        # Plot Confusion Matrix
        plotconfusion(confusion_matrix, classes=tester.classes,
                      xlabel='True Class', ylabel='Predicted Class')

    print("Finishing time: {}".format(time.asctime()))

