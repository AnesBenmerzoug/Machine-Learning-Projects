import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from .imageTransform import ImageTransform
from .model import MNIST_Network
import random


class MNISTTester:
    def __init__(self, parameters):
        self.params = parameters
        self.classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

        # Transform applied to each image
        transform = transforms.Compose(
            [transforms.ToTensor(), ImageTransform(self.params)]
        )

        # Initialize datasets
        self.testset = MNIST(
            root=self.params.dataset_dir,
            train=False,
            download=True,
            transform=transform,
        )

        # Initialize loaders
        self.testloader = DataLoader(
            self.testset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.num_workers,
        )

        # Checking for GPU
        self.use_gpu = self.params.use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")

        # Load Trained Model
        path = self.params.model_dir / "trained_model.pt"
        self.model = self.load_model(path, self.use_gpu)
        self.model.to(self.device)

        print(self.model)

        print("Number of parameters = {}".format(self.model.num_parameters()))

    def test_model(self):
        self.model.eval()
        correct = 0
        total = 0
        confusion_matrix = [[0] * 10 for i in range(10)]
        class_correct = [0] * 10
        class_total = [0] * 10
        for data in self.testloader:
            # Split data tuple
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # Forward step
            outputs = self.model(inputs)
            _, guesses = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (guesses == labels.data).sum()
            c = (guesses == labels.data).squeeze()
            for j in range(int(inputs.size(0))):
                label = labels[j]
                guess_i = guesses[j]
                class_correct[label.data.item()] += c[j]
                class_total[label.data.item()] += 1
                confusion_matrix[label.data.item()][guess_i] += 1
        total_accuracy = correct * 1.0 / total * 100.0
        class_accuracy = [
            class_correct[k] * 1.0 / class_total[k] * 100.0
            if class_total[k] != 0
            else 0.0
            for k in range(10)
        ]
        confusion_matrix = [
            [
                confusion_matrix[i][j] / class_total[i]
                for j in range(len(confusion_matrix[i]))
            ]
            for i in range(len(confusion_matrix))
        ]
        return total_accuracy, class_accuracy, confusion_matrix

    def test_random_sample(self):
        print("Testing random sample")
        self.model.eval()
        dataiter = iter(self.testloader)
        data = []
        for k in range(20):
            data.append(dataiter.next())
        random.shuffle(data)
        images, labels = data[random.randint(0, len(data) - 1)]
        print(
            "GroundTruth: ",
            " ".join(
                "%5s" % self.classes[labels[j]] for j in range(int(images.size(0)))
            ),
        )
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = self.model(images)
        _, predicted = torch.max(outputs.data, 1)
        print(
            "Predicted:     ",
            " ".join(
                "%5s" % self.classes[predicted[j]] for j in range(int(images.size(0)))
            ),
        )

    def load_model(self, path, use_gpu=False):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        return MNIST_Network.load_model(package, self.params, use_gpu)
