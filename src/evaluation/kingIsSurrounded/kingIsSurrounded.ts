import {Board, getThroneSq, Piece} from "@/board";
import {getBottomNeighbor, getLeftNeighbor, getRightNeighbor, getTopNeighbor} from "@/board/utils";

const THRONE_SQ = getThroneSq();

export const kingIsSurrounded = (board: Board) => {
  const {kingSq} = board;

  if (board.kingSq === -1) {
    throw new Error("King square is not defined");
  }

  return [
    getLeftNeighbor(kingSq),
    getTopNeighbor(kingSq),
    getRightNeighbor(kingSq),
    getBottomNeighbor(kingSq)
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
