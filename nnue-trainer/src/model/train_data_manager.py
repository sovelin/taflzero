from abc import abstractmethod


class TrainDataManager:
    @abstractmethod
    def save_bin_data(self, writer, fen: str, wdl: float):
        pass

    @abstractmethod
    def calculate_nnue_input_layer(self, fen: str):
        pass

    @abstractmethod
    def get_packed_size(self):
        pass

    @abstractmethod
    def get_record_size(self):
        pass

    @abstractmethod
    def get_bin_folder(self):
        pass

    @abstractmethod
    def parse_record(self, record: bytes):
        pass

    @abstractmethod
    def get_input_size(self):
        pass