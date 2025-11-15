import struct
from enum import Enum

from torch import tensor, float32

from src.networks.hnefatafl.constants import NETWORK_SIZE
from src.model.train_data_manager import TrainDataManager
from src.networks.hnefatafl.set_fen import FenCalculator, SQS_COUNT
from src.utils import unpack_bits, pack_bits

class Piece(Enum):
    ATTACKER = 0
    DEFENDER = 1
    KING = 2

def calculate_nnue_index(piece: Piece, square: int):
    pieces_mapper = {
        Piece.ATTACKER: 0,
        Piece.DEFENDER: 1,
        Piece.KING: 2
    }

    return pieces_mapper[piece] * SQS_COUNT + square

class HnefataflNetworkDataManager(TrainDataManager):
    fen_calculator = FenCalculator()

    def calculate_nnue_input_layer(self, fen: str):
        return self.fen_calculator.calculate_nnue_input_layer(fen)

    def get_packed_size(self):
        return (NETWORK_SIZE + 7) // 8

    def get_record_size(self):
        return self.get_packed_size() + 1 * 4

    def parse_record(self, record: bytes):
        packed_size = self.get_packed_size()
        packed_input = record[:self.get_packed_size()]
        wdl_score = struct.unpack('f', record[packed_size:packed_size + 4])[0]
        nnue_input = unpack_bits(packed_input, NETWORK_SIZE)

        return (
            [tensor(nnue_input, dtype=float32)],
            tensor(wdl_score, dtype=float32)
        )

    def get_bin_folder(self):
        return "hnefatafl"

    def save_bin_data(self, writer, fen: str, wdl: float):
        nnue_input = self.calculate_nnue_input_layer(fen)
        packed_input = pack_bits(nnue_input)
        writer.write(packed_input)
        writer.write(struct.pack('f', wdl))
