# pylint: disable=missing-class-docstring,invalid-name
import torch
import torch.nn.functional as F

class Cifar10CNN(torch.nn.Module):

    def __init__(self):
        super(Cifar10CNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(64)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv5 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(128)
        self.conv6 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = torch.nn.BatchNorm2d(128)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2)

        self.fc1 = torch.nn.Linear(128 * 4 * 4, 128)

        self.softmax = torch.nn.Softmax()
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x): # pylint: disable=missing-function-docstring
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool1(x)

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool2(x)

        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.pool3(x)

        x = x.view(-1, 128 * 4 * 4)

        x = self.fc1(x)
        x = self.softmax(self.fc2(x))

        return x
