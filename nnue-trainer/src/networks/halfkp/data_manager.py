import struct

import numpy as np
from torch import tensor, float32

from src.networks.halfkp.constants import NETWORK_INPUT_SIZE
from src.model.train_data_manager import TrainDataManager
from src.utils import ctzll, unpack_bits, pack_bits



def calculate_nnue_index(color: bool, piece: int, square: int, king_square: int):
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
    }

    piece_index = pieces_mapper[piece] * 2 + colors_mapper[color]

    return square + (piece_index + king_square * 10) * 64

class HalfKPDataManager(TrainDataManager):
    def calculate_nnue_input_layer(self, fen: str):
        board = chess.Board(fen)

        nnue_input_us = np.zeros(NETWORK_INPUT_SIZE, dtype=np.int8)
        nnue_input_them = np.zeros(NETWORK_INPUT_SIZE, dtype=np.int8)

        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)

        occupied = board.occupied & ~board.kings

        while occupied:
            square = ctzll(occupied)
            piece = board.piece_at(square)
            color = piece.color
            piece_type = piece.piece_type
            nnue_input_us[calculate_nnue_index(color, piece_type, square, white_king_square)] = 1
            nnue_input_them[calculate_nnue_index(not color, piece_type, square ^ 56, black_king_square ^ 56)] = 1

            occupied &= occupied - 1

        return nnue_input_us, nnue_input_them

    def get_unit_size(self):
        return NETWORK_INPUT_SIZE // 8

    def get_packed_size(self):
        return 2 * self.get_unit_size()

    def get_record_size(self):
        return self.get_packed_size() + 4

    def parse_record(self, record: bytes):
        packed_input = record[:self.get_packed_size()]
        eval_score = struct.unpack('f', record[self.get_packed_size():])[0]
        unit_size = self.get_unit_size()
        nnue_input1 = unpack_bits(packed_input[:unit_size], NETWORK_INPUT_SIZE)
        nnue_input2 = unpack_bits(packed_input[unit_size:], NETWORK_INPUT_SIZE)

        return (
            [
                tensor(nnue_input1, dtype=float32),
                tensor(nnue_input2, dtype=float32)
            ],
            tensor(eval_score, dtype=float32),
        )

    def get_bin_folder(self):
        return "halfkp"

    def save_bin_data(self, writer, fen: str, eval_score: float):
        nnue_input1, nnue_input2 = self.calculate_nnue_input_layer(fen)
        packed_input1 = pack_bits(nnue_input1)
        packed_input2 = pack_bits(nnue_input2)
        writer.write(packed_input1)
        writer.write(packed_input2)
        writer.write(struct.pack('f', eval_score))
