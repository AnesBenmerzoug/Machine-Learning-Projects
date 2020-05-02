import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import NLLLoss
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from .imageTransform import ImageTransform
from .model import MNIST_Network
from torch.nn.utils import clip_grad_norm
from collections import namedtuple
import time
import os


class MNISTTrainer:
    def __init__(self, parameters):
        self.params = parameters

        # Transform applied to each image
        transform = transforms.Compose(
            [transforms.ToTensor(), ImageTransform(self.params)]
        )

        # Initialize datasets
        self.trainset = MNIST(
            root=self.params.datasetDir, train=True, download=True, transform=transform
        )
        self.testset = MNIST(
            root=self.params.datasetDir, train=False, download=True, transform=transform
        )

        # Initialize loaders
        self.trainloader = DataLoader(
            self.trainset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.num_workers,
            sampler=RandomSampler(self.trainset),
        )

        self.testloader = DataLoader(
            self.testset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.num_workers,
        )

        # Checking for GPU
        self.useGPU = self.params.useGPU and torch.cuda.is_available()

        # Initialize model
        if self.params.resumeTraining is False:
            print("Training New Model")
            self.model = MNIST_Network(self.params)
        else:
            print("Resuming Training")
            self.load_model(self.useGPU)

        print(self.model)

        print("Number of parameters = {}".format(self.model.num_parameters()))

        # Checking for GPU
        self.useGPU = self.params.useGPU and torch.cuda.is_available()
        if self.useGPU is True:
            print("Using GPU")
            try:
                self.model.cuda()
            except RuntimeError:
                print("Failed to find GPU. Using CPU instead")
                self.useGPU = False
                self.model.cpu()
            except UserWarning:
                print("GPU is too old. Using CPU instead")
                self.useGPU = False
                self.model.cpu()
        else:
            print("Using CPU")

        # Setup optimizer
        self.optimizer = self.optimizer_select()

        # Criterion
        self.criterion = NLLLoss()

    def train_model(self):
        max_accuracy = None
        best_model = None
        avg_losses = np.zeros(self.params.num_epochs)
        for epoch in range(self.params.num_epochs):
            print("Epoch {}".format(epoch + 1))

            print("Learning Rate= {}".format(self.optimizer.param_groups[0]["lr"]))

            # Set mode to training
            self.model.train()

            # Go through the training set
            avg_losses[epoch] = self.train_epoch()

            print("Average loss= {:.3f}".format(avg_losses[epoch]))

            # Switch to eval and go through the test set
            self.model.eval()

            # Go through the test set
            test_accuracy = self.test_epoch()
            print(
                "In Epoch {}, Obtained Accuracy {:.2f}".format(epoch + 1, test_accuracy)
            )
            if max_accuracy is None or max_accuracy < test_accuracy:
                max_accuracy = test_accuracy
                best_model = self.model.state_dict()
        # Saving trained model
        self.save_model(best_model, max_accuracy)
        return avg_losses

    def train_epoch(self):
        losses = 0.0
        for batch_index, (data) in enumerate(self.trainloader, 1):
            if batch_index % 200 == 0:
                print("Step {}".format(batch_index))
                print("Average Loss so far: {}".format(losses / batch_index))
            # Split data tuple
            inputs, labels = data
            # Wrap it in Variables
            if self.useGPU is True:
                inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            # Main Model Forward Step
            output = self.model(inputs)
            # Loss Computation
            loss = self.criterion(output, labels)
            inf = float("inf")
            if loss.data.item() == inf or loss.data.item() == -inf:
                print("Warning, received inf loss. Skipping it")
            elif loss.data.item() != loss.data.item():
                print("Warning, received nan loss.")
            else:
                losses = losses + loss.data.item()
            # Zero the optimizer gradient
            self.optimizer.zero_grad()
            # Backward step
            loss.backward()
            # Clip gradients
            clip_grad_norm(self.model.parameters(), self.params.max_norm)
            # Weight Update
            self.optimizer.step()
            if self.useGPU is True:
                torch.cuda.synchronize()
            del inputs, labels, data, loss, output
        # Compute the average loss for this epoch
        avg_loss = losses / len(self.trainloader)
        return avg_loss

    def test_epoch(self):
        correct = 0
        total = 0
        for data in self.testloader:
            # Split data tuple
            inputs, labels = data
            # Wrap it in Variables
            if self.useGPU is True:
                inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            # Forward step
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size()[0]
            correct += torch.sum(predicted == labels.data)
            del outputs, inputs, labels, data
        total_accuracy = correct * 1.0 / total * 100.0
        return total_accuracy

    def save_model(self, model_parameters, model_accuracy):
        self.model.load_state_dict(model_parameters)
        torch.save(
            self.serialize(),
            os.path.join(
                self.params.savedModelDir,
                "Trained_Model_{}".format(int(model_accuracy))
                + "_"
                + time.strftime("%d.%m.20%y_%H.%M"),
            ),
        )

    def load_model(self, useGPU=False):
        package = torch.load(
            self.params.trainedModelPath, map_location=lambda storage, loc: storage
        )
        self.model = MNIST_Network.load_model(package, useGPU)
        parameters = package["params"]
        self.params = namedtuple("Parameters", (parameters.keys()))(
            *parameters.values()
        )
        self.optimizer = self.optimizer_select()

    def serialize(self):
        model_is_cuda = next(self.model.parameters()).is_cuda
        model = self.model.cpu() if model_is_cuda else self.model
        package = {
            "state_dict": model.state_dict(),
            "params": self.params._asdict(),
            "optim_dict": self.optimizer.state_dict(),
        }
        return package

    def optimizer_select(self):
        if self.params.optimizer == "Adam":
            return optim.Adam(self.model.parameters(), lr=self.params.learning_rate)
        elif self.params.optimizer == "Adadelta":
            return optim.Adadelta(self.model.parameters(), lr=self.params.learning_rate)
        elif self.params.optimizer == "SGD":
            return optim.SGD(
                self.model.parameters(),
                lr=self.params.learning_rate,
                momentum=self.params.momentum,
                nesterov=self.params.nesterov,
            )
        else:
            raise NotImplementedError()
