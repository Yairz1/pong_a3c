import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

def plot_reward(time_lst, reward_lst):
    x = []
    y = []
    for i in range(len(time_lst[:])):
        if time_lst[i] == i and i > 0:
            break
        x.append(time_lst[i] / 60)
        y.append(reward_lst[i])

    # np.save('data-.npy', np.array([np.asarray(x), np.asarray(y)]))
    plt.plot(x, y)
    plt.xlabel("time (min)")
    plt.ylabel("rewards")
    plt.savefig('Graph.jpeg')
    plt.show()


def plot_all():

    all_plot = np.asarray([np.load('data/data_adam_linear.npy'),
                           np.load('data/data_adam_lstm.npy'),
                           np.load('data/data_adam_original.npy'),
                           np.load('data/data_rms_linear.npy'),
                           np.load('data/data_rms_lstm.npy'),
                           np.load('data/data_rms_original.npy')])

    legends = np.asarray(['ADAM_Linear',
                          'ADAM_LSTM',
                          'ADAM_Original',
                          'RMS_Linear',
                          'RMS_LSTM',
                          'RMS_Original'])

    fig, axs = plt.subplots(2, 3)
    plt.setp(axs, xlim=(-1, 18), ylim=(-22, 22))
    a = 0
    b = 0
    for j in range(len(all_plot[:])):
        time_lst = all_plot[j][0]
        reward_lst = all_plot[j][1]
        x = []
        y = []

        for i in range(len(time_lst[:])):
            if time_lst[i] == i and i > 0:
                break
            x.append(time_lst[i] / 60)
            y.append(reward_lst[i])
        axs[a, b].plot(x, y)
        axs[a, b].set_title(legends[j])

        b += 1
        if b == 3:
            a += 1
            b = 0

    for ax in axs.flat:
        ax.set(xlabel='time (min)', ylabel='rewards')

    fig.suptitle('Pong A3C', fontsize=16)
    fig.savefig('data/Graph.jpeg')
    plt.show()


def simulate(env, model, num_episode=20, max_episode=int(1.2e5)):
    model.eval()
    AVG = 0
    for i_episode in range(num_episode):
        obs = env.reset()
        obs = torch.from_numpy(obs)
        cx = torch.zeros(1, 256)
        hx = torch.zeros(1, 256)
        G = 0
        for t in range(max_episode):
            with torch.no_grad():
                env.render()
                v_t, prob, (hx, cx) = model((obs.unsqueeze(0), (hx, cx)))
                P_t = F.softmax(prob, dim=-1)
                action = P_t.multinomial(num_samples=1).detach()
                obs, reward, done, info = env.step(action)
                obs = torch.from_numpy(obs)
                G += reward
                if done:
                    break
        AVG += G
    print(AVG / num_episode)
    env.close()