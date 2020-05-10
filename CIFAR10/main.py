import faulthandler
from pathlib import Path
import time

import click

from src import *


class Parameters:
    # Model Parameters
    image_size = [32, 32]
    conv1_in_size = 3
    conv1_out_size = 64
    conv2_out_size = 128
    conv3_out_size = 256
    conv4_out_size = 512
    fc1_out_size = 1024
    fc2_out_size = 256
    output_size = 10
    model_dir = Path("./trained_models/")
    # Dataset Parameters
    dataset_dir = Path("./data/")
    num_workers = 2
    # Training Parameters
    num_epochs = 200
    batch_size = 128
    max_norm = 400
    # Optimizer Parameters
    optimizer = "Adam"
    learning_rate = 1.0e-3
    momentum = 0.9
    nesterov = True


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
        trainer = CIFAR10Trainer(Parameters)
        # Training the model
        avg_losses = trainer.train_model()
        # Plot losses
        plotlosses(
            avg_losses,
            title="Average Loss per Epoch",
            xlabel="Epoch",
            ylabel="Average Loss",
        )

    # Instantiating the test
    tester = CIFAR10Tester(Parameters)
    # Testing the model
    tester.test_random_sample()
    # Testing the model accuracy
    total_accuracy, class_accuracy, confusion_matrix = tester.test_model()

    print("Total Accuracy = {:.2f}".format(total_accuracy))
    for i in range(len(class_accuracy)):
        print("Accuracy for class {}: {:.2f}".format(i, class_accuracy[i]))

    # Plot Per Class Accuracy
    plotaccuracy(
        class_accuracy,
        classes=tester.classes,
        title="Classification Accuracy per Class",
        xlabel="Class",
        ylabel="Accuracy",
    )

    # Plot Confusion Matrix
    plotconfusion(
        confusion_matrix,
        classes=tester.classes,
        xlabel="True Class",
        ylabel="Predicted Class",
    )

    print("Finishing time: {}".format(time.asctime()))


if __name__ == "__main__":
    main()
