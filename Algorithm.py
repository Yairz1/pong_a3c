import copy
import threading
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import threading

from Network import Net, ActorCritic
from envs import create_atari_env
from preprocess import state_process
import sys


def flush_print(str):
    print(str, end="")
    sys.stdout.flush()


def policy(p, action_space):
    return np.random.choice(action_space, 1, p=p.detach().numpy().reshape(-1))[0]


class A3C:
    def __init__(self, **args):
        '''
        :param env:
        :param policy:
        :param model:
        :param action_space:
        :param T:
        :param lock:
        :param t_max:
        :param T_max:
        :param gamma:
        :param optimizer:
        :param entropy_coef:
        :param gae_lambda:
        '''
        self.global_model = args['model']
        self.action_space = args['action_space']
        self._lock = args['lock']
        self.T = args['T']
        self.env = args['env']
        self.policy = args['policy']
        self.t_max = args['t_max']
        self.T_max = args['T_max']
        self.gamma = args['gamma']
        self.optimizer = args['optimizer']
        self.entropy_coef = args['entropy_coef']
        self.gae_lambda = args['gae_lambda']

    def _async_step(self, local_model):
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=50)
        for local_p, global_p in zip(local_model.parameters(),
                                     self.global_model.parameters()):
            if global_p.grad is not None:
                return
            global_p._grad = local_p.grad
        self.optimizer.step()

    def actor_critic(self):
        env = create_atari_env(self.env)

        t = 0
        s_t = env.reset()
        local_model = ActorCritic(env.observation_space.shape[0], env.action_space)
        done = True
        while True:

            with self._lock:
                if self.T.value >= self.T_max: break

            if done:
                cx = torch.zeros(1, 256)
                hx = torch.zeros(1, 256)
            else:
                cx = cx.detach()
                hx = hx.detach()

            local_model.load_state_dict(self.global_model.state_dict())
            t_start = t

            episode_info = []
            values = []
            total_reward = 0
            while t - t_start < self.t_max:
                v_t, logit, (hx, cx) = local_model((s_t,(hx, cx)))
                P_t = F.softmax(logit, dim=-1)
                log_P_t = F.log_softmax(logit, dim=-1)
                a_t = self.policy(P_t, self.action_space)
                s_t_1, r_t, done, _ = env.step(a_t)
                r_t = max(min(r_t, 1), -1)
                total_reward += r_t
                t += 1
                with self._lock:
                    self.T.value += 1
                episode_info.append((a_t, s_t, r_t, P_t,log_P_t))
                values.append(v_t)
                s_t = s_t_1

                if done:
                    break

            R = 0 if done else local_model((s_t,(hx, cx)))[0]
            values.append(R)
            policy_loss = 0
            value_loss = 0
            gae = torch.zeros(1, 1)
            for i in reversed(range(len(episode_info))):
                a_i, s_i, r_i, P_i, log_P_i = episode_info[i]
                R = r_i + self.gamma * R
                A = R - values[i]
                value_loss = value_loss + 0.5 * A.pow(2)

                H = -(log_P_i * P_i).sum(1, keepdim=True)
                # Generalized Advantage Estimation
                delta_t = r_i + self.gamma * values[i + 1] - values[i]
                gae = gae * self.gamma * self.gae_lambda + delta_t
                policy_loss = policy_loss - log_P_i[0][a_i] * gae.detach() - self.entropy_coef * H

            self.optimizer.zero_grad()
            local_model.zero_grad()
            J = policy_loss + 0.5 * value_loss
            J.backward()
            flush_print(
                f'\r process id {threading.get_ident()} loss:{J.detach().numpy()[0][0]}, training process: {round(100 * self.T.value / self.T_max)}%')
            self._async_step(local_model)
            if done:
                s_t = env.reset()
