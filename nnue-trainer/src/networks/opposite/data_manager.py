import struct
from functools import lru_cache

import numpy as np
from torch import tensor, float32

from src.model.train_data_manager import TrainDataManager
from src.networks.opposite.constants import NETWORK_INPUT_SIZE
from src.utils import ctzll, unpack_bits, pack_bits

SCALE = 400

@lru_cache(maxsize=1024)
def calculate_nnue_index(color: bool, piece: int, square: int):
    colors_mapper = {
        chess.WHITE: 0,
        chess.BLACK: 1
    }

    pieces_mapper = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    return 64 * 6 * colors_mapper[color] + pieces_mapper[piece] * 64 + square

class OppositeNetworkDataManager(TrainDataManager):
    # constructor
    def __init__(self):
        super().__init__()

        self.packed_size = self.get_packed_size()

    @lru_cache(maxsize=1000000)
    def calculate_nnue_input_layer(self, fen: str):
        board = chess.Board(fen)
        nnue_input_us = np.zeros(NETWORK_INPUT_SIZE, dtype=np.int8)
        nnue_input_them = np.zeros(NETWORK_INPUT_SIZE, dtype=np.int8)
        occupied = board.occupied

        while occupied:
            square = ctzll(occupied)
            piece = board.piece_at(square)
            color = piece.color
            piece_type = piece.piece_type
            nnue_input_us[calculate_nnue_index(color, piece_type, square)] = 1
            nnue_input_them[calculate_nnue_index(not color, piece_type, square ^ 56)] = 1
            occupied &= occupied - 1

        turn = 1 if board.turn == chess.WHITE else 0

        return nnue_input_us, nnue_input_them, turn

    def get_packed_size(self):
        return 2 * self.get_unit_size()

    def get_record_size(self):
        return self.get_packed_size() + 3 * 4

    def get_unit_size(self):
        return NETWORK_INPUT_SIZE // 8

    def parse_record(self, record: bytes):
        # packed_size = self.get_packed_size()
        packed_input = record[:self.packed_size]
        eval_score, wdl_score, stm = struct.unpack('fff', record[packed_size:packed_size + 12])
        # eval_score = struct.unpack('f', record[self.packed_size:self.packed_size + 4])[0]
        # wdl_score = struct.unpack('f', record[self.packed_size + 4:self.packed_size + 8])[0]
        # stm = struct.unpack('f', record[self.packed_size + 8:self.packed_size + 12])[0]
        nnue_input_us = unpack_bits(packed_input[:self.get_unit_size()], NETWORK_INPUT_SIZE)
        nnue_input_them = unpack_bits(packed_input[self.get_unit_size():], NETWORK_INPUT_SIZE)

        return (
            [tensor(nnue_input_us, dtype=float32)],
            [tensor(nnue_input_them, dtype=float32)],
            tensor(eval_score, dtype=float32),
            tensor(wdl_score, dtype=float32),
            tensor(stm, dtype=float32)
        )

    def get_bin_folder(self):
        return "opposite"

    def save_bin_data(self, writer, fen: str, eval_score: float, wdl: float):
        nnue_input_us, nnue_input_them, stm = self.calculate_nnue_input_layer(fen)
        packed_input_us = pack_bits(nnue_input_us)
        packed_input_them = pack_bits(nnue_input_them)
        writer.write(packed_input_us)
        writer.write(packed_input_them)
        writer.write(struct.pack('f', eval_score))
        writer.write(struct.pack('f', wdl))
        writer.write(struct.pack('f', stm))
