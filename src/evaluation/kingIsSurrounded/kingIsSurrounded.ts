import {Board, THRONE_SQ, Piece, VERTICAL_HORIZONTAL_NEIGHBORS} from "@/board";

export const kingIsSurrounded = (board: Board) => {
  const {kingSq} = board;

  if (board.kingSq === -1) {
    throw new Error("King square is not defined");
  }

  return VERTICAL_HORIZONTAL_NEIGHBORS[kingSq]
    .filter((sq) => {
    if (sq === Piece.EMPTY) {
      return false;
    }

    if (sq === THRONE_SQ) {
     return true;
    }

    return board.board[sq!] === Piece.ATTACKER;
  }).length === 4;
}
