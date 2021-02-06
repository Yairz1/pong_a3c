import time
import matplotlib.pyplot as plt
import gym
import torch

import my_optim
from Algorithm import A3C
from Network import Net
from envs import create_atari_env
from preprocess import state_process
from Algorithm import policy


def simulate(env, model, action_space):
    for i_episode in range(20):
        obs = env.reset()
        for t in range(400):
            with torch.no_grad():
                env.render()
                P_i, _ = model(state_process(obs))
                action = policy(P_i, action_space)
                obs, reward, done, info = env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
    env.close()


def train():
    env = create_atari_env('PongDeterministic-v4')
    global_model = Net(env.action_space.n)
    global_model.share_memory()
    gamma = 0.99
    optimizer = my_optim.SharedAdam(global_model.parameters(), lr=0.0001)
    optimizer.share_memory()
    t_max = 5
    T_max = 100000
    entropy_coef = 0.01
    gae_lambda = 1
    num_processes = 8
    a3c = A3C(env,
              policy,
              global_model,
              env.action_space.n,
              t_max,
              T_max,
              gamma,
              optimizer,
              entropy_coef,
              gae_lambda,
              num_processes)
    model = a3c.multi_actor_critic()

    simulate(env, model, env.action_space.n)
    env.close()


if __name__ == '__main__':
    train()
