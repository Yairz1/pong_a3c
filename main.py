from __future__ import print_function
import argparse
import os

import torch
import torch.multiprocessing as mp

import my_optim
from Algorithm import A3C
from envs import create_atari_env
from Network import ActorCritic
from test import test

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae-lambda', type=float, default=1.00,
                    help='lambda parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='PongDeterministic-v4',
                    help='environment to train on (default: PongDeterministic-v4)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')


if __name__ == '__main__':
    mp.set_start_method('spawn')

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    env = create_atari_env(args.env_name)
    shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

    T = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, T))
    p.start()
    processes.append(p)
    ############
    t_max = -1
    T_max = -1

    ############
    for rank in range(0, args.num_processes):
        a3c = A3C(shared_model, 6, lock, T, args.env_name, t_max, T_max, args.gamma, optimizer, args.entropy_coef,
                 args.gae_lambda, args.seed, args.num_steps, args.max_episode_length)
        p = mp.Process(target=a3c.actor_critic, args=(rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()








































# import os
# import threading
# import time
# import matplotlib.pyplot as plt
# import gym
# import torch
# import torch.multiprocessing as mp
# import my_optim
# from Algorithm import A3C
# from Network import Net, ActorCritic
# from envs import create_atari_env
# from preprocess import state_process
# from Algorithm import policy
# import torch.nn.functional as F
#
# from test import test
#
#
# class counter:
#     def __init__(self):
#         self.value = 0
#
#
# def simulate(env, model, action_space):
#     env = create_atari_env(env)
#     model.eval()
#     AVG = 0
#     for i_episode in range(20):
#         obs = env.reset()
#         cx = torch.zeros(1, 256)
#         hx = torch.zeros(1, 256)
#         G = 0
#         for t in range(400):
#             with torch.no_grad():
#                 env.render()
#                 v_t, logit, (hx, cx) = model((obs, (hx, cx)))
#                 P_t = F.softmax(logit, dim=-1)
#
#                 action = policy(P_t, action_space)
#                 obs, reward, done, info = env.step(action)
#                 G += reward
#                 if done:
#                     print("Episode finished after {} timesteps".format(t + 1))
#                     break
#         AVG += G
#     print(AVG / 20)
#     env.close()
#
#
# def train():
#     os.environ['OMP_NUM_THREADS'] = '1'
#     os.environ['CUDA_VISIBLE_DEVICES'] = ""
#     torch.manual_seed(1)
#     env = create_atari_env('PongDeterministic-v4')
#     # global_model = Net(env.action_space.n)
#     global_model = ActorCritic(env.observation_space.shape[0], env.action_space)
#
#     global_model.share_memory()
#
#     optimizer = my_optim.SharedAdam(global_model.parameters(), lr=0.0001)
#     optimizer.share_memory()
#     num_processes = 5
#     T = mp.Value('i', 0)
#     lock = mp.Lock()
#     args = {'env': 'PongDeterministic-v4',
#             'policy': policy,
#             'model': global_model,
#             'action_space': 6,
#             'T': T,
#             'lock': lock,
#             't_max': 20,
#             'T_max': 100000 * 20,
#             'gamma': 0.99,
#             'optimizer': optimizer,
#             'entropy_coef': 0.01,
#             'gae_lambda': 1,
#             'seed': 1,
#             'num_step': 10000 * 20,
#             'max_episode_length': 20}
#     processes = []
#     p = mp.Process(target=test, args=(num_processes, args, global_model, T))
#     p.start()
#     processes.append(p)
#     for rank in range(0, num_processes):
#         a3c = A3C(**args)
#         p = mp.Process(target=a3c.actor_critic, args=(rank,))
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()
#
#     simulate('PongDeterministic-v4', global_model, args['action_space'])
#
#
# if __name__ == '__main__':
#     mp.set_start_method('spawn')
#     train()
