from torch import clamp, nn, no_grad
import torch
import numpy as np


from src.networks.hnefatafl.constants import NETWORK_SIZE
from src.model.nnue import NNUE

SCALE = 400
QA = 255
QB = 64

class HnefataflNetwork(NNUE):
    def __init__(self, hidden_size):
        super(HnefataflNetwork, self).__init__()
        self.fc1 = nn.Linear(NETWORK_SIZE, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, 1, bias=False)
        self.relu = nn.ReLU()

    def save_weights(self, epoch: int, train_directory: str):
        super().save_weights(epoch, train_directory)
        self._save_weight(self.fc1, "fc1", epoch, train_directory)
        self._save_weight(self.fc2, "fc2", epoch, train_directory)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def print_weights_like_rust(self):
        print("===== NNUE WEIGHTS (RUST FORMAT) =====")

        # ---------- Layer 1 ----------
        w1 = self.fc1.weight.detach().cpu().numpy()  # [HIDDEN, INPUTS]

        print("\n-- FC1 (input-major: [363][32]) --")
        for i in range(w1.shape[1]):      # INPUT index
            for h in range(w1.shape[0]):  # HIDDEN index
                v = w1[h, i]
                if v != 0:
                    print(f"W1[{i}][{h}] = {v}")

        # ---------- Layer 2 ----------
        w2 = self.fc2.weight.detach().cpu().numpy().reshape(-1)  # [HIDDEN]

        print("\n-- FC2 (hidden-major: [32]) --")
        for h in range(len(w2)):
            v = w2[h]
            if v != 0:
                print(f"W2[{h}] = {v}")

        # acc print

        print("===== END NNUE WEIGHTS =====")



    def print_acc_like_rust(self, x):
        """
            x — вектор входа длиной 363 (tensor float32, 0/1), как из manager.calculate_nnue_input_layer
            Печатает квантованный acc[h] и eval в тех же единицах, что Rust.
            """
        with no_grad():
            # 1) приводим x к 1D numpy
            if x.dim() > 1:
                x = x.view(-1)
            x_np = x.detach().cpu().numpy()  # [INPUTS]

            # 2) веса
            w1 = self.fc1.weight.detach().cpu().numpy()     # [HIDDEN, INPUTS]
            w2 = self.fc2.weight.detach().cpu().numpy().reshape(-1)  # [HIDDEN]

            # 3) скрытый слой (до ReLU, float)
            # h_float[h] = sum_i w1[h, i] * x[i]
            h_float = w1 @ x_np  # [HIDDEN]

            # 4) квантованный acc, как в Rust: acc[h] ≈ round(h_float[h] * QA)
            acc_q = np.round(h_float * QA).astype(int)  # [HIDDEN]

            # 5) квантованные W2, как в Rust
            w2_q = np.round(w2 * QB).astype(int)  # [HIDDEN]

            # 6) печать как в Rust'овском print_weights
            print("\n-- ACCUMULATORS (quantized, like Rust) --")
            for h in range(len(acc_q)):
                if acc_q[h] != 0:
                    print(f"Acc[{h}] = {acc_q[h]}")

            # 7) считаем sum и eval в точности как в Rust
            relu_acc_q = np.maximum(acc_q, 0)
            sum_q = int(np.sum(relu_acc_q * w2_q))

            eval_rs = (sum_q * SCALE) // (QA * QB)

            print(f"\nNNUE sum (Python, should match Rust): {sum_q}")
            print(f"Eval (Python Rust-style) = {eval_rs}")