import copy
import threading

import torch

from preprocess import state_process


def policy(p):
    return p.argmax()


def num2action(num):
    action_map = {0: 0, 1: 2, 2: 5}
    return action_map[int(num)]


class A3C:
    def __init__(self, env, policy, model, action_space, t_max, T_max, gamma, optimizer):
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

        while True:
            with self._lock:
                if self.T >= self.T_max: break
            local_model = copy.deepcopy(self.global_model)
            t_start = t
            done = False
            episode_info = []
            while not done and t - t_start < self.t_max:
                P_t, v_t = local_model(state_process(s_t))
                a_t = self.policy(P_t)
                s_t_1, r_t, done, _ = self.env.step(num2action(a_t))
                t += 1
                with self._lock:
                    self.T += 1
                episode_info.append((a_t, s_t, r_t, P_t, v_t))
                s_t = s_t_1
            R = 0 if done else v_t
            policy_loss = 0
            value_loss = 0
            H = 0
            for i in range(t - 1, t_start - 1, -1):
                a_i, s_i, r_i, P_i, v_i = episode_info[i - t_start]
                R = r_i + self.gamma * R
                A = R - v_i

                policy_loss += A * torch.log(P_i[0][a_i])
                value_loss += 0.5 * A.pow(2)
                H = 0

            self.optimizer.zero_grad()
            (policy_loss + value_loss + H).backward()
            self._async_step(local_model)
            if done: s_t = self.env.reset()
