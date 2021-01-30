import copy
import threading

import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    def __init__(self, action_space):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc = nn.Linear(32 * 24 * 18, 256)
        self.value_fc = nn.Linear(256, 1)
        self.actions_fc = nn.Linear(256, action_space)

    # x represents observation which is RGB image.
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.relu(x)

        # Apply softmax as policy and fc as value function.
        output1 = F.softmax(self.actions_fc(x), dim=1)
        output2 = self.value_fc(x)
        return output1, output2


class A3C:
    def __init__(self, env, model, action_space):
        self.global_model = model
        self.action_space = action_space
        self._lock = threading.Lock()
        self.T = 0
        self.env = env

    def actor_critic(self):
        t = 1
        while True:
            thread_model = copy.deepcopy(self.global_model)
            d_theta = 0
            t_start = t
            state = self.env.reset()
            while True:
                break

        with self._lock:
            self.T += 1
