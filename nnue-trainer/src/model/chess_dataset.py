import csv
import os

from torch.utils.data import IterableDataset, get_worker_info
from src.model.train_data_manager import TrainDataManager


class ChessDataset(IterableDataset):
    def __init__(self, file_path, train_data_manager: TrainDataManager, positions_count: int):
        print("Prepare dataset...")
        self.data_manager = train_data_manager
        self.positions_count = positions_count
        self.input_size = train_data_manager.get_input_size()
        self.dataset_positions_count = 0
        self.packed_size = train_data_manager.get_packed_size()
        self.record_size = train_data_manager.get_record_size()
        self.bin_file_path = ".".join([train_data_manager.get_bin_folder(), file_path, "bin"])
        print("Bin file path: ", self.bin_file_path)

        if os.path.exists(self.bin_file_path):
            print("Dataset already prepared")
            self.dataset_positions_count = os.path.getsize(self.bin_file_path) // self.record_size
            return

        with (open(file_path, 'r') as f):
            reader = csv.reader(f)
            print('reader')

            with open(self.bin_file_path, 'wb') as writer:
                for idx, row in enumerate(reader):
                    if idx % 10000 == 0:
                        print(f"Processed positions: {idx}", flush=True)
                    fen, wdl = row
                    try:
                        self.data_manager.save_bin_data(writer, fen, float(wdl))
                        self.dataset_positions_count += 1
                    except Exception as e:
                        print('lol')
                        print(e.with_traceback())
                        continue
        print("Dataset prepared")

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            start = 0
            step = 1
        else:
            start = worker_info.id
            step = worker_info.num_workers

        with open(self.bin_file_path, 'rb') as f:
            for idx in range(start, self.dataset_positions_count, step):
                if idx >= self.positions_count:
                    break

                f.seek(idx * self.record_size)
                record = f.read(self.record_size)

                if not record:
                    break

                yield self.data_manager.parse_record(record)
