import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, action_space):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
        self.bn1 = torch.nn.BatchNorm2d(num_features=16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(num_features=32)

        self.fc = nn.Linear(32 * 24 * 18, 256) #
        self.value_fc = nn.Linear(256, 1)
        self.actions_fc = nn.Linear(256, action_space)

    # x represents observation which is RGB image.
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.relu(x)

        # Apply softmax as policy and fc as value function.
        actor = F.softmax(self.actions_fc(x), dim=1)
        critic = self.value_fc(x)
        return actor, critic
