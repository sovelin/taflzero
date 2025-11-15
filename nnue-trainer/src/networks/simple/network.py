import torch
from torch import clamp, nn
from torch.nn import functional as F


from src.networks.simple.constants import SIMPLE_NETWORK_INPUT_SIZE
from src.model.nnue import NNUE

SCALE = 400
QA = 255
QB = 64

class SimpleNetwork(NNUE):
    def __init__(self, hidden_size):
        super(SimpleNetwork, self).__init__()
        self.fc1 = nn.Linear(SIMPLE_NETWORK_INPUT_SIZE, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, 1, bias=False)
        self.relu = nn.ReLU()

    def save_weights(self, epoch: int, train_directory: str):
        super().save_weights(epoch, train_directory)
        self._save_weight(self.fc1, "fc1", epoch, train_directory)
        self._save_weight(self.fc2, "fc2", epoch, train_directory)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = clamp(x, 0, 1)
        x = self.fc2(x)
        return x


class SimpleDeepNetwork(NNUE):
    def __init__(self, hidden_size, second_hidden_size):
        super(SimpleDeepNetwork, self).__init__()
        self.fc1 = nn.Linear(SIMPLE_NETWORK_INPUT_SIZE, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, second_hidden_size, bias=False)
        self.fc3 = nn.Linear(second_hidden_size, 1, bias=False)
        self.relu = nn.ReLU()

    def save_weights(self, epoch: int, train_directory: str):
        super().save_weights(epoch, train_directory)

        self._save_weight(self.fc1, "fc1", epoch, train_directory)
        self._save_weight(self.fc2, "fc2", epoch, train_directory)
        self._save_weight(self.fc3, "fc3", epoch, train_directory)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

        x = F.linear(x, self.fc1.weight * QA)
        x = self.relu(x)
        x /= QA
        x = F.linear(x, self.fc2.weight * QA)
        # x = clamp(x, 0, QA)
        x = self.relu(x)
        x = F.linear(x, self.fc3.weight * QB)
        print(x)
        x /= (QB * QA)
        return x
