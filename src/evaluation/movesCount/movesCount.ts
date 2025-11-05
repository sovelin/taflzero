import {Board, COL, ROW} from "@/board";
import {POSSIBLE_MOVES_COUNT} from "@/moves/attackers";

export const getPossibleMovesCount = (board: Board, sq: number): number => {
  const row = ROW[sq]
  const col = COL[sq]

  const rowOcc = board.rowOcc[row];
  const horizontalMoves = POSSIBLE_MOVES_COUNT[col][rowOcc];

  const colOcc = board.colOcc[col];
  const verticalMoves = POSSIBLE_MOVES_COUNT[row][colOcc]

  return horizontalMoves + verticalMoves;
}
