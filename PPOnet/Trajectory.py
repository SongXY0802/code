import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trajectory:
    def __init__(self, instance, ins_id):
        self.instance = instance
        self.ins_id = ins_id

        self.values = []
        self.rewards = []
        self.td_target = []
        self.td_delta = []

    def save(self, reward, value):
        self.rewards.append(reward)
        self.values.append(value)

    def advantage(self, gamma, lmbda):
        dones = np.ones(len(self.rewards))
        dones[len(self.rewards) - 1] = 0
        for i in range(len(self.rewards)):
            td_target_i = self.rewards[i] + gamma * self.values[i + 1] * dones[i]
            td_delta_i = td_target_i - self.values[i]
            self.td_target.append(td_target_i.item())
            self.td_delta.append(td_delta_i.item())
        advantage_list = []
        advantage = 0.0
        for delta in self.td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        self.advantage = advantage_list
