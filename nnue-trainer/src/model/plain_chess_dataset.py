import csv

import torch
from torch import tensor, float32
from torch.utils.data import IterableDataset

from src.model.train_data_manager import TrainDataManager


class PlainChessDataset(IterableDataset):
    def __init__(self, file_path, train_data_manager: TrainDataManager, positions_count):
        self.file_path = file_path
        self.train_data_manager = train_data_manager
        self.positions_count = positions_count

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            start = 0
            step = 1
        else:
            # Split workload among workers
            start = worker_info.id
            step = worker_info.num_workers

        with open(self.file_path, 'r') as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if idx >= self.positions_count:
                     break

                if idx % step == start:
                    fen, score, wdl = row
                    score = float(score)
                    wdl = float(wdl)
                    # try:
                    inputs1, inputs2, stm = self.train_data_manager.calculate_nnue_input_layer(fen)
                    # print(inputs1, inputs2, stm)

                    yield (
                        [tensor(inputs1, dtype=float32)],
                        [tensor(inputs2, dtype=float32)],
                        tensor(score, dtype=float32),
                        tensor(wdl, dtype=float32),
                        tensor(stm, dtype=float32)
                    )
                    #except Exception as e:
                    #    print(e)
                    #    continue