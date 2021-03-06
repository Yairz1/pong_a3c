import time
from collections import deque

import torch
import torch.nn.functional as F
from envs import create_atari_env
import numpy as np


def test(rank, args, model_constructor, shared_model, T, time_list, reward_list):
    """
    This method runs on a separate process every {x} seconds and print the reward.
    :param rank:
    :param args:
    :param model_constructor:
    :param shared_model:
    :param T:
    :param time_list:
    :param reward_list:
    :return:
    """
    torch.manual_seed(args.seed + rank)
    env = create_atari_env('PongDeterministic-v4')
    env.seed(args.seed + rank)
    model = model_constructor(env.observation_space.shape[0], env.action_space)
    model.eval()
    i = 0
    state = env.reset()
    reward_sum = 0
    done = True

    start_time = time.time()
    actions = deque(maxlen=100)
    episode_length = 0
    while T.value < args.T_max:
        state = torch.from_numpy(state)
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()

        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            total_time = time.time() - start_time
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(total_time)),
                T.value, T.value / total_time,
                reward_sum, episode_length))
            if i < 100:
                time_list[i] = int(total_time)
                reward_list[i] = reward_sum
                i += 1
                np.save('data_rms_lstm.npy', np.array([np.asarray(time_list), np.asarray(reward_list)]))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(30)
