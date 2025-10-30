import {Board, getThroneSq, Piece} from "@/board";
import {getBottomSquare, getLeftSquare, getRightSquare, getTopSquare} from "@/board/utils";

const THRONE_SQ = getThroneSq();

export const kingIsSurrounded = (board: Board) => {
  const {kingSq} = board;

  if (board.kingSq === -1) {
    throw new Error("King square is not defined");
  }

  return [
    getLeftSquare(kingSq),
    getTopSquare(kingSq),
    getRightSquare(kingSq),
    getBottomSquare(kingSq)
  ].filter((sq) => {
    if (sq === Piece.EMPTY) {
      return false;
    }

    if (sq === THRONE_SQ) {
     return true;
    }

    return board.board[sq!] === Piece.ATTACKER;
  }).length === 4;
}
