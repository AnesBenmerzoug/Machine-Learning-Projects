import torch
import numpy as np
import torch.optim as optim
from torch.nn import NLLLoss
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.nn.utils import clip_grad_norm
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from src.imageTransform import ImageTransform
from src.model import CIFAR10_Network
from src.optimizer import SVRG
from collections import namedtuple
from copy import deepcopy
import time
import os


class CIFAR10Trainer(object):
    def __init__(self, parameters):
        self.params = parameters

        # Transform applied to each image
        transform = transforms.Compose([transforms.ToTensor(), ImageTransform(self.params)])

        # Initialize datasets
        self.trainset = CIFAR10(root=self.params.datasetDir, train=True,
                                download=True, transform=transform)
        self.testset = CIFAR10(root=self.params.datasetDir, train=False,
                               download=True, transform=transform)

        # Initialize loaders
        self.trainloader = DataLoader(self.trainset, batch_size=self.params.batch_size,
                                      shuffle=False, num_workers=self.params.num_workers,
                                      sampler=RandomSampler(self.trainset))

        self.testloader = DataLoader(self.testset, batch_size=self.params.batch_size,
                                     shuffle=False, num_workers=self.params.num_workers)

        # Checking for GPU
        self.useGPU = self.params.useGPU and torch.cuda.is_available()

        # Initialize model
        if self.params.resumeTraining is False:
            print("Training New Model")
            self.model = CIFAR10_Network(self.params)
        else:
            print("Resuming Training")
            self.load_model(self.useGPU)

        if self.params.optimizer == 'SVRG':
            self.snapshot_model = deepcopy(self.model)

        print(self.model)

        print("Number of parameters = {}".format(self.model.num_parameters()))

        if self.useGPU is True:
            print("Using GPU")
            try:
                self.model.cuda()
                if self.params.optimizer == 'SVRG':
                    self.snapshot_model.cuda()
            except RuntimeError:
                print("Failed to find GPU. Using CPU instead")
                self.useGPU = False
                self.model.cpu()
                if self.params.optimizer == 'SVRG':
                    self.snapshot_model.cpu()
            except UserWarning:
                print("GPU is too old. Using CPU instead")
                self.useGPU = False
                self.model.cpu()
                if self.params.optimizer == 'SVRG':
                    self.snapshot_model.cpu()
        else:
            print("Using CPU")

        # Setup optimizer
        self.optimizer = self.optimizer_select()

        # Criterion
        self.criterion = NLLLoss()

    def snapshot_closure(self):
        def closure(data, target):
            # Wrap it in Variables
            if self.useGPU is True:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            # Forward step
            output = self.snapshot_model(data)
            # Loss computation
            snapshot_loss = self.criterion(output, target)
            # Zero the optimizer gradient
            self.optimizer.zero_grad()
            # Backward step
            snapshot_loss.backward()
            # Clip gradients
            clip_grad_norm(self.snapshot_model.parameters(), self.params.max_norm)
            return snapshot_loss
        return closure

    def train_model(self):
        max_accuracy = None
        best_model = None
        avg_losses = np.zeros(self.params.num_epochs)
        for epoch in range(self.params.num_epochs):
            print("Epoch {}".format(epoch + 1))

            if self.params.optimizer == 'SVRG':
                # Update SVRG snapshot
                self.optimizer.update_snapshot(dataloader=self.trainloader, closure=self.snapshot_closure())

            print("Learning Rate= {}".format(self.optimizer.param_groups[0]['lr']))

            # Set mode to training
            self.model.train()

            # Go through the training set
            avg_losses[epoch] = self.train_epoch()

            print("Average loss= {}".format(avg_losses[epoch]))

            # Switch to eval and go through the test set
            self.model.eval()

            # Go through the test set
            test_accuracy = self.test_epoch()
            print("In Epoch {}, Obtained Accuracy {:.2f}".format(epoch + 1, test_accuracy))
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
            if loss.data[0] == inf or loss.data[0] == -inf:
                print("Warning, received inf loss. Skipping it")
            elif loss.data[0] != loss.data[0]:
                print("Warning, received nan loss.")
            else:
                losses = losses + loss.data[0]
            # Zero the optimizer gradient
            self.optimizer.zero_grad()
            # Backward step
            loss.backward()
            # Clip gradients
            clip_grad_norm(self.model.parameters(), self.params.max_norm)
            if self.params.optimizer == 'SVRG':
                # Snapshot Model Forward Backward
                snapshot_output = self.snapshot_model(inputs)
                snapshot_loss = self.criterion(snapshot_output, labels)
                self.snapshot_model.zero_grad()
                snapshot_loss.backward()
                clip_grad_norm(self.snapshot_model.parameters(), self.params.max_norm)
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
        for (data) in self.testloader:
            # Split data tuple
            inputs, labels = data
            # Wrap it in Variables
            if self.useGPU is True:
                inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            # Forward step
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += torch.sum(predicted == labels.data)
            del outputs, inputs, labels, data
        total_accuracy = correct / total * 100.0
        return total_accuracy

    def save_model(self, model_parameters, model_accuracy):
        self.model.load_state_dict(model_parameters)
        torch.save(self.serialize(),
                   os.path.join(self.params.savedModelDir, 'Trained_Model_{}'.format(int(model_accuracy))
                                + '_' + time.strftime("%d.%m.20%y_%H.%M")))

    def load_model(self, useGPU=False):
        package = torch.load(self.params.trainedModelPath, map_location=lambda storage, loc: storage)
        self.model = CIFAR10_Network.load_model(package, useGPU)
        parameters = package['params']
        self.params = namedtuple('Parameters', (parameters.keys()))(*parameters.values())
        self.optimizer = self.optimizer_select()

    def serialize(self):
        model_is_cuda = next(self.model.parameters()).is_cuda
        model = self.model.cpu() if model_is_cuda else self.model
        package = {
            'state_dict': model.state_dict(),
            'params': self.params._asdict(),
            'optim_dict': self.optimizer.state_dict()
        }
        return package

    def optimizer_select(self):
        if self.params.optimizer == 'Adam':
            return optim.Adam(self.model.parameters(), lr=self.params.learning_rate)
        elif self.params.optimizer == 'Adadelta':
            return optim.Adadelta(self.model.parameters(), lr=self.params.learning_rate)
        elif self.params.optimizer == 'SGD':
            return optim.SGD(self.model.parameters(), lr=self.params.learning_rate,
                             momentum=self.params.momentum, nesterov=self.params.nesterov)
        elif self.params.optimizer == 'SVRG':
            return SVRG(self.model.parameters(), self.snapshot_model.parameters(),
                        lr=self.params.learning_rate, momentum=self.params.momentum,
                        nesterov=self.params.nesterov, update_frequency=self.params.update_frequency)
        else:
            raise NotImplementedError
