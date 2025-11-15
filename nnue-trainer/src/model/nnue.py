from abc import abstractmethod

import torch
from torch import nn


def save_layer_weights(weights: nn.Linear, weights_filename, biases_filename):
    weight_matrix = weights.weight.cpu().data.numpy()  # shape [out_features, in_features]
    # biases_vector = weights.bias.cpu().data.numpy()  # shape [out_features]
    flat_weights = weight_matrix.flatten()  # shape [out_features * in_features]
    # flat_biases = biases_vector.flatten()  # shape [out_features]

    with open(weights_filename, 'w') as file:
        file.write(','.join(str(x) for x in flat_weights))
        file.write('\n')

    # with open(biases_filename.replace('weights', 'biases'), 'w') as file:
    #     file.write(','.join(str(x) for x in flat_biases))
    #     file.write('\n')


class NNUE(nn.Module):

    def _save_weight(self, layer: nn.Linear, name: str, epoch: int, train_directory: str):
        save_layer_weights(
            layer,
            f"{train_directory}/{name}.{epoch}.weights.csv",
            f"{train_directory}/{name}.{epoch}.biases.csv"
        )

    def save_weights(self, epoch: int, train_directory: str):
        model = self.state_dict()
        torch.save(model, f"{train_directory}/model.{epoch}.pth")

    def load_weights(self, epoch: int, train_directory: str):
        model = torch.load(f"{train_directory}/model.{epoch}.pth", weights_only=True)
        self.load_state_dict(model)
