import torch
import numpy as np
import torch.optim as optim
from torch.nn import NLLLoss, MSELoss
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.nn.utils import clip_grad_norm
from torchvision.datasets import STL10
from torchvision.transforms import transforms
from src.model import STL10_Network
from collections import namedtuple
import time
import os


class STL10Trainer(object):
    def __init__(self, parameters):
        self.params = parameters

        # Transform applied to each image
        transform = transforms.ToTensor()

        # Initialize datasets
        self.unlabelledset = STL10(root=self.params.datasetDir, split='unlabeled',
                                  download=True, transform=transform)
        self.trainset = STL10(root=self.params.datasetDir, split='train',
                              download=True, transform=transform)
        self.testset = STL10(root=self.params.datasetDir, split='test',
                             download=True, transform=transform)

        # Initialize loaders
        self.unlabelledloader = DataLoader(self.unlabelledset, batch_size=self.params.batch_size,
                                           shuffle=False, num_workers=self.params.num_workers,
                                           sampler=RandomSampler(self.unlabelledset))
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
            self.model = STL10_Network(self.params)
        else:
            print("Resuming Training")
            self.load_model(self.useGPU)

        print(self.model)

        print("Number of parameters = {}".format(self.model.num_parameters()))

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

        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=self.params.step_size,
                                                   gamma=self.params.decay_coeff)

        # Criterion
        self.criterion_supervised = NLLLoss()
        self.criterion_unsupervised = MSELoss()

    def train_model(self):
        max_accuracy = None
        best_model = None
        avg_losses = np.zeros(self.params.num_epochs_unlabelled + self.params.num_epochs_labelled)
        # Unsupervised learning first
        for epoch in range(self.params.num_epochs_unlabelled + self.params.num_epochs_labelled):
            print("Epoch {}".format(epoch + 1))

            print("Learning Rate= {}".format(self.optimizer.param_groups[0]['lr']))

            # Set mode to training
            self.model.train()

            if epoch < self.params.num_epochs_unlabelled:
                # Unsupervised training
                avg_losses[epoch] = self.train_epoch_unsupervised()
            else:
                # Supervised training
                avg_losses[epoch] = self.train_epoch_supervised()

            print("In Epoch {}, Average loss= {}".format(epoch + 1, avg_losses[epoch]))

            if epoch < self.params.num_epochs_unlabelled:
                continue

            # Update learning rate
            self.scheduler.step(epoch)

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

    def train_epoch_unsupervised(self):
        losses = 0.0
        for batch_index, (data) in enumerate(self.unlabelledloader, 1):
            if batch_index % 50 == 0:
                print("Step {}".format(batch_index))
                print("Average Loss so far: {}".format(losses / batch_index))
            # Split data tuple
            inputs, _ = data
            # Wrap it in Variables
            if self.useGPU is True:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            # Main Model Forward Step
            _, rec_outputs = self.model(inputs, unsupervised=True)
            # Loss Computation
            loss = None
            for output_rec, output in rec_outputs:
                if loss is None:
                    loss = self.params.lambda_rec * self.criterion_unsupervised(output_rec, inputs)
                else:
                    loss = loss + self.params.lambda_mid * self.criterion_unsupervised(output_rec, output.detach())
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
            # Weight Update
            self.optimizer.step()
            if self.useGPU is True:
                torch.cuda.synchronize()
            del inputs, data, loss, rec_outputs
        # Compute the average loss for this epoch
        avg_loss = losses / len(self.unlabelledloader)
        return avg_loss

    def train_epoch_supervised(self):
        losses = 0.0
        for batch_index, (data) in enumerate(self.trainloader, 1):
            if batch_index % 50 == 0:
                print("Step {}".format(batch_index))
                print("Average Loss so far: {}".format(losses / batch_index))
            # Split data tuple
            inputs, labels = data
            # Wrap it in Variables
            if self.useGPU is True:
                inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            # Main Model Forward Step
            output, rec_outputs = self.model(inputs)
            # Loss Computation
            loss = self.criterion_supervised(output, labels)
            rec_loss = None
            for out_rec, out in rec_outputs:
                if rec_loss is None:
                    rec_loss = self.params.lambda_rec * self.criterion_unsupervised(out_rec, inputs)
                else:
                    rec_loss = rec_loss + self.params.lambda_mid * self.criterion_unsupervised(out_rec, out.detach())
            loss = loss + rec_loss
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
            outputs, _ = self.model(inputs)
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
        self.model = STL10_Network.load_model(package, useGPU)
        self.optimizer = self.optimizer_select()
        parameters = package['params']
        self.params = namedtuple('Parameters', (parameters.keys()))(*parameters.values())

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
        else:
            raise NotImplementedError
