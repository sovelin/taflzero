from torch import clamp, nn, cat

from src.networks.halfkp.constants import NETWORK_INPUT_SIZE
from src.model.nnue import NNUE


class HalfKPNetwork(NNUE):
    def __init__(self, hidden_size):
        super(HalfKPNetwork, self).__init__()
        self.fc1_us = nn.Linear(NETWORK_INPUT_SIZE, hidden_size, bias=False)
        self.fc1_them = nn.Linear(NETWORK_INPUT_SIZE, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2 * hidden_size, 1, bias=False)

    def save_weights(self, epoch: int, train_directory: str):
        super().save_weights(epoch, train_directory)
        self._save_weight(self.fc1_us, "fc1_us", epoch, train_directory)
        self._save_weight(self.fc1_them, "fc1_them", epoch, train_directory)
        self._save_weight(self.fc2, "fc2", epoch, train_directory)

    def forward(self, x1, x2):
        first_input = self.fc1_us(x1)
        second_input = self.fc1_them(x2)

        acc1 = clamp(first_input, 0, 1)
        acc2 = clamp(second_input, 0, 1)

        x = cat((acc1, acc2), 1)
        return self.fc2(x)
