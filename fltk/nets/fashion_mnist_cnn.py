# pylint: disable=missing-function-docstring,missing-class-docstring,invalid-name
import torch

class FashionMNISTCNN(torch.nn.Module):

    def __init__(self):
        super(FashionMNISTCNN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(self.flatten(x))

        return x
