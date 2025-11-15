from enum import Enum
import numpy as np

BOARD_SIZE = 11
SQS_COUNT = BOARD_SIZE * BOARD_SIZE
PIECES_COUNT = 3  # attacker, defender, king


class Piece(Enum):
    ATTACKER = 0
    DEFENDER = 1
    KING = 2


# maps piece → NNUE idx
def calculate_nnue_index(piece: Piece, square: int):
    return piece.value * SQS_COUNT + square


class FenParseError(Exception):
    pass


class FenCalculator:
    def __init__(self):
        pass

    def calculate_nnue_input_layer(self, fen: str):
        current_row = 10
        current_col = 0
        pass_cnt = ''

        res = np.zeros(BOARD_SIZE * BOARD_SIZE * PIECES_COUNT, dtype=np.uint8)

        for ch in fen:
            if ch == '/':
                current_row -= 1
                current_col = 0
                pass_cnt = ''
                if current_row < 0:
                    raise FenParseError("Too many rows in FEN")
                continue

            if ch.isdigit():
                pass_cnt += ch
                continue
            if ch == ' ':
                break

            if ch in ('k', 'a', 'd'):
                if pass_cnt:
                    current_col += int(pass_cnt)
                    pass_cnt = ''

                if current_col >= BOARD_SIZE:
                    raise FenParseError("Column index out of bounds")

                square = current_row * BOARD_SIZE + current_col

                if ch == 'k':
                    piece = Piece.KING
                elif ch == 'a':
                    piece = Piece.ATTACKER
                elif ch == 'd':
                    piece = Piece.DEFENDER
                else:
                    raise FenParseError(f"Invalid piece character: {ch}")

                nnue_index = calculate_nnue_index(piece, square)
                res[nnue_index] = 1

                current_col += 1


        return res