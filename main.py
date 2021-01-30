import time
import matplotlib.pyplot as plt
import gym
import torch

from Algorithm import Net
from preprocess import state_process

env = gym.make('PongDeterministic-v4')
ob = env.reset()
net = Net(action_space=3)
x = net(state_process(ob))
for _ in range(1000):
    env.render()
    time.sleep(.05)
    ob, reward, _, _ = env.step(env.action_space.sample())
    if reward:
        print(reward)
        plt.imshow(ob)
        plt.show()
env.close()
