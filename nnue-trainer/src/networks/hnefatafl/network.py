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
    

    def manual_eval_float(self, x):
        """
        Полностью ручной пересчёт NNUE в float:
        - acc первого слоя
        - после ReLU
        - вклад второго слоя
        - финальный вывод (без квантования)
        """

        # --- x → numpy 1D ---
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        x = x.reshape(-1)

        w1 = self.fc1.weight.detach().cpu().numpy()
        w2 = self.fc2.weight.detach().cpu().numpy().reshape(-1)

        INPUTS = len(x)
        HIDDEN = len(w2)

        # ---------- 1. ACC первого слоя (float) ----------
        acc = []
        print("\n=== ACC (float, до ReLU) ===")
        for h in range(HIDDEN):
            s = 0.0
            for i in range(INPUTS):
                if x[i] != 0:                 # как в Rust set_input()
                    s += w1[h, i] * x[i]
            acc.append(s)
            print(f"Acc[{h}] = {s}")

        # ---------- 2. После ReLU ----------
        print("\n=== ReLU(ACC) ===")
        relu_acc = []
        for h, val in enumerate(acc):
            r = max(0.0, val)
            relu_acc.append(r)
            print(f"ReLU[{h}] = {r}")

        # ---------- 3. Второй слой ----------
        print("\n=== Вклад второго слоя (relu * w2) ===")
        contributions = []
        sum_out = 0.0
        for h in range(HIDDEN):
            c = relu_acc[h] * w2[h]
            contributions.append(c)
            sum_out += c
            print(f"h={h}: relu={relu_acc[h]}, w2={w2[h]}, contrib={c}, running_sum={sum_out}")

        # ---------- 4. Итоговый вывод ----------
        print(f"\n=== NNUE FLOAT OUTPUT ===")
        print(f"Output = {sum_out}")

        return {
            "acc": acc,
            "relu": relu_acc,
            "contrib": contributions,
            "output": sum_out
        }


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

    def print_w2_debug(self):
        print("\n--- PYTHON W2 FLOAT ---")
        w2 = self.fc2.weight.detach().cpu().numpy().reshape(-1)
        for i, v in enumerate(w2):
            print(f"W2_float[{i}] = {v}")

        w2_q = np.round(w2 * QB).astype(int)
        print("\n--- PYTHON W2 QUANTIZED (*64) ---")
        for i, v in enumerate(w2_q):
            print(f"W2_q[{i}] = {v}")

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
    

    def debug_eval_python(self, x):
        """
        Полный эквивалент Rust debug_eval().
        Повторяет каждый шаг Rust: печать ACC, суммирование, eval, входы.
        """

        # ----------------------------
        # 0. Подготовка входа
        # ----------------------------
        if x.dim() > 1:
            x = x.view(-1)

        x = x.detach().cpu().numpy()  # [INPUTS]
        INPUTS = len(x)

        # ----------------------------
        # 1. Вытягиваем веса
        # ----------------------------
        w1 = self.fc1.weight.detach().cpu().numpy()  # [HIDDEN, INPUTS]
        w2 = self.fc2.weight.detach().cpu().numpy().reshape(-1)  # [HIDDEN]
        HIDDEN = len(w2)

        # ----------------------------
        # 2. Полностью пересчитываем acc[] как делает Rust new()
        # ----------------------------
        acc = [0] * HIDDEN

        for h in range(HIDDEN):
            sum_h = 0.0
            for i in range(INPUTS):
                if x[i] != 0:               # как Rust: set_input()
                    sum_h += w1[h][i]
            acc[h] = int(round(sum_h * QA))  # QA = 255

        # ----------------------------
        # 3. Печатаем ACC (точно как Rust)
        # ----------------------------
        print("-- ACC (Python) --")
        for h in range(HIDDEN):
            print(f"Acc[{h}] = {acc[h]}")

        # ----------------------------
        # 4. NNUE sum (Rust logic)
        # ----------------------------
        sum_q = 0
        for h in range(HIDDEN):
            xh = acc[h] if acc[h] > 0 else 0                 # max(0)
            wh = int(round(w2[h] * QB))                      # QB = 64
            sum_q += xh * wh
            print(f"NNUE sum: {sum_q}, x: {xh}, w: {wh}")

        print(f"NNUE sum: {sum_q}")

        # ----------------------------
        # 5. Финальная оценка (Rust)
        # ----------------------------
        num = sum_q * SCALE           # SCALE = 400
        den = QA * QB                 # 255 * 64
        eval_rs = num // den

        print(f"NNUE eval = {eval_rs}")

        # ----------------------------
        # 6. Печать входов όπως Rust
        # ----------------------------
        print("-- INPUTS (Python) --")
        for i in range(INPUTS):
            if x[i] != 0:
                print(f"Input {i} is set")



     # -------------------
    # ⭐ DEBUG: PRINT INPUT INDEXES LIKE RUST
    # -------------------
    def print_inputs_like_rust(self, x):
        """
        Печатает индексы входов, которые включены (== 1),
        чтобы сравнить их с Rust NNUE inputs.
        """
        with no_grad():
            if x.dim() > 1:
                x = x.view(-1)

            x_np = x.detach().cpu().numpy()

            print("\n--- PYTHON INPUT INDEXES (non-zero == active) ---")
            for i, v in enumerate(x_np):
                if v != 0:
                    print(i)
            print("--- END PYTHON INPUT INDEXES ---")
    
    def print_weights(self):
        print("===== NNUE WEIGHTS =====")

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

        print("===== END NNUE WEIGHTS =====")