from __future__ import print_function
import argparse
import os
import torch
import torch.multiprocessing as mp
from my_optim import get_optim
from Algorithm import A3C
from envs import create_atari_env
from Network import get_model
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
                    help='max grad norm (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=10,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--t-max', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1.5e5,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--T-max', type=int, default=8e6,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='PongDeterministic-v4',
                    help='environment to train on (default: PongDeterministic-v4)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--model', default='lstm',
                    help='default lstm/linear/original.')
parser.add_argument('--optimization', default='rms',
                    help='default adam/rms.')






if __name__ == '__main__':

    # plot_all()

    mp.set_start_method('spawn')

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    args = parser.parse_args()
    model_constructor = get_model(args.model)
    torch.manual_seed(args.seed)
    env = create_atari_env(args.env_name)
    shared_model = model_constructor(env.observation_space.shape[0], env.action_space)

    # shared_model.load_state_dict(torch.load("data/Weights"))
    # shared_model.eval()

    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        opti_constructor = get_optim(args.optimization)
        optimizer = opti_constructor(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

    T = mp.Value('i', 0)
    lock = mp.Lock()
    time_lst = mp.Array('d', range(100))
    reward_lst = mp.Array('d', range(100))

    p = mp.Process(target=test,
                   args=(args.num_processes, args, model_constructor, shared_model, T, time_lst, reward_lst))
    p.start()
    processes.append(p)
    ############

    for rank in range(0, args.num_processes):
        a3c = A3C(model_constructor, shared_model, 6, lock, T, args.env_name, args.t_max, args.T_max, args.gamma,
                  optimizer,
                  args.entropy_coef,
                  args.gae_lambda, args.seed, args.max_episode_length)
        p = mp.Process(target=a3c.actor_critic, args=(rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    torch.save(shared_model.state_dict(), "data/Weights")
    simulate(env, shared_model, 3)
    plot_reward(time_lst, reward_lst)
