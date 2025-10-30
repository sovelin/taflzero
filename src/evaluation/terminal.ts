import {Board, getBottomLeftSquare, getBottomRightSquare, getTopLeftSquare, getTopRightSquare, Side} from "@/board";
import {kingIsSurrounded} from "@/evaluation/kingIsSurrounded/kingIsSurrounded";
import {printBoard} from "@/board/print";

const cornersSquares = [
  getTopRightSquare(),
  getTopLeftSquare(),
  getBottomLeftSquare(),
  getBottomRightSquare(),
];

export const checkTerminal = (board: Board, shouldPrint = false): Side | null => {
  if (cornersSquares.includes(board.kingSq) || board.attackersCount < 2) {
    return Side.DEFENDERS;
  }

  if (board.attackersCount < 2) {
    return Side.DEFENDERS;
  }

  if (kingIsSurrounded(board)) {
    return Side.ATTACKERS;
  }

  return null;
}
