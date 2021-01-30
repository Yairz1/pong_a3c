import torch


def state_process(state):
    state = torch.FloatTensor(state).permute(2, 0, 1)
    return state.unsqueeze(0)
