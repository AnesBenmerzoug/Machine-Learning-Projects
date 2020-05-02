import faulthandler
import time

from src import *


class Parameters:
    image_size = [28, 28]
    hidden_size = 16
    output_size = 10
    batch_size = 32
    savedModelDir = "./trained_models/"
    testModelPath = "./trained_models/Trained_Model_99_Accuracy"
    # Dataset Parameters
    datasetDir = "./data/"
    num_workers = 2
    # Training Parameters
    trainModel = False
    useGPU = False
    resumeTraining = False
    num_epochs = 1000
    max_norm = 400
    # Optimizer Parameters
    optimizer = "Adam"
    learning_rate = 1.0e-2
    momentum = 0.9
    nesterov = True
    update_frequency = 2


if __name__ == "__main__":
    print("Starting time: {}".format(time.asctime()))

    # To have a more verbose output in case of an exception
    faulthandler.enable()

    if Parameters.trainModel is True:
        # Instantiating the trainer
        trainer = MNISTTrainer(Parameters)
        # Training the model
        avg_losses = trainer.train_model()
        # Plot losses
        plotlosses(
            avg_losses,
            title="Average Loss per Epoch",
            xlabel="Epoch",
            ylabel="Average Loss",
        )

    else:
        # Instantiating the test
        tester = MNISTTester(Parameters)
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
