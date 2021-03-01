from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """LeNet-like network for tests with MNIST (28x28)."""

    def __init__(self, in_channels=1, num_classes=10, **kwargs):
        super().__init__()
        # main part of the network
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 16, 120)
        self.fc2 = nn.Linear(120, 84)

        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(84, num_classes)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc(out)
        return out
