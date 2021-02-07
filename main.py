import os
import threading
import time
import matplotlib.pyplot as plt
import gym
import torch
import torch.multiprocessing as mp
import my_optim
from Algorithm import A3C
from Network import Net, ActorCritic
from envs import create_atari_env
from preprocess import state_process
from Algorithm import policy
import torch.nn.functional as F


class counter:
    def __init__(self):
        self.value = 0


def simulate(env, model, action_space):
    model.eval()
    for i_episode in range(20):
        obs = env.reset()
        cx = torch.zeros(1, 256)
        hx = torch.zeros(1, 256)
        for t in range(400):
            with torch.no_grad():
                env.render()
                v_t, logit, (hx, cx) = model((obs,(hx, cx)))
                P_t = F.softmax(logit, dim=-1)

                action = policy(P_t, action_space)
                obs, reward, done, info = env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
    env.close()


def train():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    torch.manual_seed(1)

    env = create_atari_env('PongDeterministic-v4')
    # global_model = Net(env.action_space.n)
    global_model = ActorCritic(env.observation_space.shape[0], env.action_space)

    global_model.share_memory()

    optimizer = my_optim.SharedAdam(global_model.parameters(), lr=0.0001)
    optimizer.share_memory()
    num_processes = 10
    T = mp.Value('i', 0)
    lock = mp.Lock()
    args = {'env': env,
            'policy': policy,
            'model': global_model,
            'action_space': env.action_space.n,
            'T': T,
            'lock': lock,
            't_max': 5,
            'T_max': 100000,
            'gamma': 0.99,
            'optimizer': optimizer,
            'entropy_coef': 0.01,
            'gae_lambda': 1}
    processes = []
    for rank in range(0, num_processes):
        a3c = A3C(**args)
        p = mp.Process(target=a3c.actor_critic)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    simulate(env, global_model, env.action_space.n)
    env.close()


if __name__ == '__main__':
    # mp.set_start_method('spawn')
    train()
