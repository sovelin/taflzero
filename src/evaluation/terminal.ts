import {
  Board,
  Side,
  CORNERS_SQ
} from "@/board";
import {kingIsSurrounded} from "@/evaluation/kingIsSurrounded/kingIsSurrounded";
import {defendersIsSurrounded} from "@/evaluation/defendersIsSurrounded";
import {checkFort} from "@/evaluation/checkFort";

export const checkTerminal = (board: Board): Side | null => {
  if (board.repTable.get(board.zobrist)! >= 3) {
    return Side.ATTACKERS;
  }

  if (CORNERS_SQ.includes(board.kingSq) || board.attackersCount < 2) {
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
