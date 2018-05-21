import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from src.imageTransform import ImageTransform
from src.model import MNIST_Network
from src.utils import imgshow
from collections import namedtuple
import random


class MNISTTester(object):
    def __init__(self, parameters):
        self.params = parameters
        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

        # Transform applied to each image
        transform = transforms.Compose([transforms.ToTensor(), ImageTransform(self.params)])

        # Initialize datasets
        self.testset = MNIST(root=self.params.datasetDir, train=False,
                             download=True, transform=transform)

        # Initialize loaders
        self.testloader = DataLoader(self.testset, batch_size=self.params.batch_size,
                                     shuffle=False, num_workers=self.params.num_workers)

        # Checking for GPU
        self.useGPU = self.params.useGPU and torch.cuda.is_available()

        # Load Trained Model
        self.load_model()

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

    def test_model(self):
        self.model.eval()
        correct = 0
        total = 0
        class_correct = [0]*10
        class_total = [0]*10
        for (data) in self.testloader:
            # Split data tuple
            inputs, labels = data
            # Wrap it in Variables
            if self.useGPU is True:
                inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            # Forward step
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
            c = (predicted == labels.data).squeeze()
            for j in range(int(inputs.size(0))):
                label = labels[j]
                class_correct[label.data[0]] += c[j]
                class_total[label.data[0]] += 1
        total_accuracy = correct / total * 100.0
        class_accuracy = [class_correct[k] / class_total[k] * 100.0 if class_total[k] != 0 else 0.0 for k in range(10)]
        return total_accuracy, class_accuracy

    def test_random_sample(self):
        print("Testing random sample")
        self.model.eval()
        dataiter = iter(self.testloader)
        data = []
        for k in range(20):
            data.append(dataiter.next())
        random.shuffle(data)
        images, labels = data[random.randint(0, len(data) - 1)]
        print('GroundTruth: ', ' '.join('%5s' % self.classes[labels[j]] for j in range(int(images.size(0)))))
        images, labels = Variable(images), Variable(labels)
        outputs = self.model(images)
        _, predicted = torch.max(outputs.data, 1)
        print('Predicted:     ', ' '.join('%5s' % self.classes[predicted[j]] for j in range(int(images.size(0)))))

    def load_model(self, useGPU=False):
        package = torch.load(self.params.testModelPath, map_location=lambda storage, loc: storage)
        self.model = MNIST_Network.load_model(package, useGPU)
        parameters = package['params']
        self.params = namedtuple('Parameters', (parameters.keys()))(*parameters.values())

