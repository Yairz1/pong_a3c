import numpy as np
import torch
import torch.nn.functional as F
import threading
from envs import create_atari_env
import sys


def flush_print(str):
    print(str, end="")
    sys.stdout.flush()


def policy(p, action_space):
    return np.random.choice(action_space, 1, p=p.detach().numpy().reshape(-1))[0]


class A3C:
    def __init__(self, model_constructor, global_model, action_space, lock, T, env, t_max, T_max, gamma, optimizer,
                 entropy_coef,
                 gae_lambda, seed, max_episode_length):
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
        self.global_model = global_model
        self.action_space = action_space
        self._lock = lock
        self.T = T
        self.env = env
        self.policy = policy
        self.t_max = t_max
        self.T_max = T_max
        self.gamma = gamma
        self.optimizer = optimizer
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        self.seed = seed
        self.t_max = t_max
        self.max_episode_length = max_episode_length
        self.model_constructor = model_constructor

    def async_step(self, model):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
        for param, shared_param in zip(model.parameters(),
                                       self.global_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def actor_critic(self, rank):
        '''
        This method is an implementation of the multi-process actor critic algorithm.
        We use almost similar notation as the original paper used:
        1. s_t: state at time t
        2. v_t: value at time t
        3. r_t: reward at time t
        4. p_t: action probability function at time t
        5. episdoe_info: a list that gather all the relevant information for the current episode.
           including (1)(2)(3)(4)
        6. model: the deep learning model that used as actor and critic.
        :param rank:
        :return: None. the result is a trained shared model which shared as pointer to the shared weights.
        '''
        torch.manual_seed(self.seed + rank)
        # each process should have is own env instance.
        env = create_atari_env(self.env)
        env.seed(self.seed + rank)
        # abstraction s.t we can construct several types of models.
        model = self.model_constructor(env.observation_space.shape[0], env.action_space)
        model.train()

        s_t = env.reset()
        # since we are using pytorch we have to use torch tensors instead of numpy arrays
        s_t = torch.from_numpy(s_t)
        done = True
        t = 0
        # T is shared counter among all the processes.
        while self.T.value < self.T_max:
            # Sync with the shared model
            model.load_state_dict(self.global_model.state_dict())
            # hidden states for lstm
            if done:
                cx = torch.zeros(1, 256)
                hx = torch.zeros(1, 256)
            else:
                cx = cx.detach()
                hx = hx.detach()

            t_start = t
            values = []
            episode_info = []

            while t - t_start < self.t_max:
                t += 1
                with self._lock:
                    self.T.value += 1

                v_t, prob, (hx, cx) = model((s_t.unsqueeze(0), (hx, cx)))
                P_t = F.softmax(prob, dim=-1)
                log_P_t = F.log_softmax(prob, dim=-1)
                # the action is chosen in stochastic way.
                a_t = P_t.multinomial(num_samples=1).detach()
                s_t, r_t, done, _ = env.step(a_t.numpy())
                r_t = max(min(r_t, 1), -1)

                if t >= self.max_episode_length:
                    done = True

                if done:
                    t = 0
                    s_t = env.reset()

                s_t = torch.from_numpy(s_t)
                values.append(v_t)
                episode_info.append((a_t, r_t, P_t, log_P_t))
                if done:
                    break

            R = torch.zeros(1, 1)
            if not done:
                value, _, _ = model((s_t.unsqueeze(0), (hx, cx)))
                R = value.detach()
            values.append(R)
            policy_loss = 0
            value_loss = 0
            Advantage_GAE = torch.zeros(1, 1)
            for i in reversed(range(len(episode_info))):
                a_i, r_i, P_i, log_P_i = episode_info[i]
                R = self.gamma * R + r_i
                Advantage = R - values[i]
                value_loss = value_loss + 0.5 * Advantage.pow(2)

                # Generalized Advantage Estimation (GAE)
                entropy = -(log_P_i * P_i).sum(1, keepdim=True)
                td_error = r_i + self.gamma * values[i + 1] - values[i]
                Advantage_GAE = Advantage_GAE * self.gamma * self.gae_lambda + td_error
                policy_loss = policy_loss - log_P_i[0][a_i] * Advantage_GAE.detach() - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            # J is our loss function.
            J = policy_loss + 0.5 * value_loss
            J.backward()
            flush_print(
                f'\r process id {threading.get_ident()} loss:{J.detach().numpy()[0][0]}, training process: {round(100 * self.T.value / self.T_max)}%')
            # preparing the weights s.t the shared weights will contain the local gradient.
            self.async_step(model)
            #
            self.optimizer.step()
