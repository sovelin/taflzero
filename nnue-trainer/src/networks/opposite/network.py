import torch
from torch import clamp, nn, cat
from torch.nn import functional as F


from src.model.nnue import NNUE
from src.networks.opposite.constants import NETWORK_INPUT_SIZE

SCALE = 400
QA = 255
QB = 64


def manually_fc2(acc1, acc2, stm, fc2):
    sum = 0

    shift = 0 if stm == 1 else 8
    print(shift)
    for index, item in enumerate(acc1):
        sum += item * fc2[shift + index]
        print(f"Relu({acc1[index]}) * {fc2[shift + index]} = {acc1[index] * fc2[shift + index]}")
        print("Sum: ", sum)

    shift = 8 if stm == 1 else 0
    for index, item in enumerate(acc2):
        sum += item * fc2[shift + index]
        # print Relu(0.878222) * 0.547996 = 0.481262
        print(f"Relu({acc2[index]}) * {fc2[shift + index]} = {acc2[index] * fc2[shift + index]}")
        print("Sum: ", sum)

    return sum

class OppositeNetwork(NNUE):
    def __init__(self, hidden_size):
        super(OppositeNetwork, self).__init__()
        # self.fc1 = nn.Linear(NETWORK_INPUT_SIZE, hidden_size, bias=False)
        # self.fc2 = nn.Linear(hidden_size, 1, bias=False)
        # self.relu = nn.ReLU()

        self.fc1_us = nn.Linear(NETWORK_INPUT_SIZE, hidden_size, bias=False)
        self.fc1_them = nn.Linear(NETWORK_INPUT_SIZE, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2 * hidden_size, 1, bias=False)

    def save_weights(self, epoch: int, train_directory: str):
        super().save_weights(epoch, train_directory)
        self._save_weight(self.fc1_us, "fc1_us", epoch, train_directory)
        self._save_weight(self.fc1_them, "fc1_them", epoch, train_directory)
        self._save_weight(self.fc2, "fc2", epoch, train_directory)

    def forward(self, x1, x2, stm):
        first_input = self.fc1_us(x1)
        second_input = self.fc1_them(x2)

        acc1 = self.relu(first_input)
        acc2 = self.relu(second_input)
        stm = torch.unsqueeze(stm, 1)

        x = (stm * torch.cat([acc1, acc2], dim=1)) + ((1 - stm) * torch.cat([acc2, acc1], dim=1))

        return self.fc2(x)
