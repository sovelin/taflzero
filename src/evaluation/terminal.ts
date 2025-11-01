import {
  Board,
  getBottomLeftSquare,
  getBottomRightSquare,
  getSquareAlg,
  getTopLeftSquare,
  getTopRightSquare,
  Side
} from "@/board";
import {kingIsSurrounded} from "@/evaluation/kingIsSurrounded/kingIsSurrounded";
import {defendersIsSurrounded} from "@/evaluation/defendersIsSurrounded";
import {checkFort} from "@/evaluation/checkFort";
import {Stats} from "@/evaluation/checkFort/checkFort";
import {printBoard} from "@/board/print";

const cornersSquares = [
  getTopRightSquare(),
  getTopLeftSquare(),
  getBottomLeftSquare(),
  getBottomRightSquare(),
];

export const checkTerminal = (board: Board): Side | null => {
  if (cornersSquares.includes(board.kingSq) || board.attackersCount < 2) {
    return Side.DEFENDERS;
  }

  if (board.attackersCount < 2) {
    return Side.DEFENDERS;
  }

  if (kingIsSurrounded(board)) {
    return Side.ATTACKERS;
  }

  if (defendersIsSurrounded(board)) {
    return Side.ATTACKERS;
  }

  if (checkFort(board)) {
    return Side.DEFENDERS;
  }

  return null;
}
