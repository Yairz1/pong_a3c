import time
import matplotlib.pyplot as plt
import gym
import torch

from Algorithm import A3C, num2action
from Network import Net
from envs import create_atari_env
from preprocess import state_process
from Algorithm import policy


def simulate(env, model):
    for i_episode in range(20):
        obs = env.reset()
        for t in range(400):
            with torch.no_grad():
                env.render()
                P_i, _ = model(state_process(obs))
                action = policy(P_i)
                obs, reward, done, info = env.step(num2action(action))
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
    env.close()


def train():
    env = create_atari_env('PongDeterministic-v4')
    action_space = 3
    global_model = Net(action_space)
    gamma = 0.99
    optimizer = torch.optim.Adam(global_model.parameters(), lr=0.1)
    t_max = 5
    T_max = 1000
    a3c = A3C(env,
              policy,
              global_model,
              action_space,
              t_max,
              T_max,
              gamma,
              optimizer)
    model = a3c.multi_actor_critic()

    simulate(env, model)
    env.close()


if __name__ == '__main__':
    train()
