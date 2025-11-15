import pytest

from set_fen import FenCalculator, calculate_nnue_index, Piece, BOARD_SIZE


def sq(s: str) -> int:
    """
    algebraic → square index (как в Rust: col + row * BOARD_SIZE)
    a1 = (0,0)
    k11 = (10,10)
    """
    col = ord(s[0]) - ord("a")
    row = int(s[1:]) - 1
    return row * BOARD_SIZE + col


def is_set(nnue, piece, algebraic):
    return nnue[calculate_nnue_index(piece, sq(algebraic))] == 1


def test_basic_position():
    fen = "11/4a6/11/11/7d3/5k5/11/1d9/8a2/11/10a d"

    c = FenCalculator()
    nnue = c.calculate_nnue_input_layer(fen)

    # attackers
    assert is_set(nnue, Piece.ATTACKER, "i3")
    assert is_set(nnue, Piece.ATTACKER, "e10")
    assert is_set(nnue, Piece.ATTACKER, "k1")

    # king
    assert is_set(nnue, Piece.KING, "f6")

    # defenders
    assert is_set(nnue, Piece.DEFENDER, "b4")
    assert is_set(nnue, Piece.DEFENDER, "h7")


def test_empty_board():
    fen = "11/11/11/11/11/11/11/11/11/11/11 a"

    c = FenCalculator()
    nnue = c.calculate_nnue_input_layer(fen)

    assert sum(nnue) == 0