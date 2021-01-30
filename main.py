import time

import gym
env = gym.make('PongDeterministic-v4')
env.reset()
for _ in range(1000):
    env.render()
    time.sleep(.1)
    env.step(env.action_space.sample()) # take a random action
env.close()