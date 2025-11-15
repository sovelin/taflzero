from enum import Enum

from src.networks.hnefatafl.constants import SQS_COUNT


class Piece(Enum):
    ATTACKER = 0
    DEFENDER = 1
    KING = 2

def calculate_nnue_index(piece: Piece, sq: int):
    pieces_mapper = {
        Piece.ATTACKER: 0,
        Piece.DEFENDER: 1,
        Piece.KING: 2
    }

    return pieces_mapper[piece] * SQS_COUNT + sq
