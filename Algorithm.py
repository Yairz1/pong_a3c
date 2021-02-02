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
        actor = F.softmax(self.actions_fc(x), dim=1)
        critic = self.value_fc(x)
        return actor, critic


class A3C:
    def __init__(self, env, policy, state_value, model, action_space, t_max, T_max, gamma):
        self.global_model = model
        self.action_space = action_space
        self._lock = threading.Lock()
        self.T = 0
        self.env = env
        self.policy = policy
        self.V = state_value
        self.t_max = t_max
        self.T_max = T_max
        self.gamma = gamma

    def actor_critic(self):
        t = 1
        s_t = self.env.reset()

        while True:
            with self._lock:
                if self.T < self.T_max: break
            thread_model = copy.deepcopy(self.global_model)
            t_start = t
            done = False
            episode_info = []
            while not done or t - t_start == self.t_max:
                a_t = self.policy(s_t)
                s_t_1, r_t, done, _ = self.env.step(a_t)
                t += 1
                with self._lock:
                    self.T += 1
                episode_info.append((a_t, s_t, r_t))
                s_t = s_t_1
            R = 0 if done else self.V(s_t)
            policy_loss = 0
            value_loss = 0
            H = 0
            for i in range(t - 1, t_start - 1, -1):
                a_i, s_i, r_i = episode_info[i]
                Pi, v_i = thread_model(s_i)
                R = r_i + self.gamma * R
                A = R - v_i

                policy_loss += A * torch.log(Pi[a_i])
                value_loss += 0.5 * A.pow(2)
                H = 0
            J = policy_loss + value_loss + H
            self._async_update(J)
            if done: s_t = self.env.reset()

    def _async_update(self, J):
        pass
