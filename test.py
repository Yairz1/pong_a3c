import time
from collections import deque

import torch
import torch.nn.functional as F

from Network import ActorCritic
from envs import create_atari_env


def test(rank, args, shared_model, T, time_list,reward_list):
    torch.manual_seed(args.seed + rank)
    env = create_atari_env('PongDeterministic-v4')
    env.seed(args.seed + rank)
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    model.eval()
    i = 0
    state = env.reset()
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
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

        # a quick hack to prevent the agent from stucking
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
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(30)
