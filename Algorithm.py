import copy
import threading
import numpy as np
import torch
import torch.nn.functional as F

from Network import Net
from preprocess import state_process
import sys


def flush_print(str):
    print(str, end="")
    sys.stdout.flush()


def policy(p, action_space):
    return np.random.choice(action_space, 1, p=p.detach().numpy().reshape(-1))[0]


class A3C:
    def __init__(self, env, policy, model, action_space, t_max, T_max, gamma, optimizer, entropy_coef, gae_lambda):
        self.global_model = model
        self.action_space = action_space
        self._lock = threading.Lock()
        self.T = 0
        self.env = env
        self.policy = policy
        self.t_max = t_max
        self.T_max = T_max
        self.gamma = gamma
        self.optimizer = optimizer
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda

    def multi_actor_critic(self):
        # for each thread, execute actor_critic
        self.actor_critic()
        return copy.deepcopy(self.global_model)

    def _async_step(self, local_model):
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=50)
        for local_p, global_p in zip(local_model.parameters(),
                                     self.global_model.parameters()):
            if global_p.grad is not None:
                return
            global_p._grad = local_p.grad
        self.optimizer.step()

    def actor_critic(self):
        t = 0
        s_t = self.env.reset()
        local_model = Net(self.env.action_space.n)
        while True:
            with self._lock:
                if self.T >= self.T_max: break
            # local_model = copy.deepcopy(self.global_model)

            local_model.load_state_dict(self.global_model.state_dict())
            t_start = t
            done = False
            episode_info = []
            values = []
            while not done and t - t_start < self.t_max:
                P_t, v_t = local_model(state_process(s_t))
                a_t = self.policy(P_t, self.action_space)
                s_t_1, r_t, done, _ = self.env.step(a_t)
                t += 1
                with self._lock:
                    self.T += 1
                episode_info.append((a_t, s_t, r_t, P_t))
                values.append(v_t)
                s_t = s_t_1
            R = 0 if done else v_t
            values.append(R)
            policy_loss = 0
            value_loss = 0
            gae = torch.zeros(1, 1)
            for i in reversed(range(len(episode_info))):
                a_i, s_i, r_i, P_i = episode_info[i]
                R = r_i + self.gamma * R
                A = R - values[i]
                value_loss = value_loss + 0.5 * A.pow(2)

                H = -(torch.log(P_i) * P_i).sum(1, keepdim=True)
                # Generalized Advantage Estimation
                delta_t = r_i + self.gamma * values[i + 1] - values[i]
                gae = gae * self.gamma * self.gae_lambda + delta_t
                policy_loss = policy_loss - torch.log(P_i)[0][a_i] * gae.detach() - self.entropy_coef * H

            self.optimizer.zero_grad()
            local_model.zero_grad()
            J = policy_loss + 0.5 * value_loss
            J.backward()
            flush_print(f'\r loss:{J.detach().numpy()[0][0]}, training process: {round(100 * self.T / self.T_max)} %')
            self._async_step(local_model)
            if done: s_t = self.env.reset()
