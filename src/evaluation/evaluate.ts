import {Board, Side} from "@/board";
import {checkTerminal} from "@/evaluation/terminal";
import {MATE_SCORE} from "@/evaluation/constants";

export const sidedEval = (board: Board, score: number): number => {
  return board.sideToMove === Side.DEFENDERS
    ? score
    : -score;
}

export const evaluateBoard = (board: Board): number => {
  // Simple evaluation function: difference in number of pieces
  const {attackersCount, defendersCount} = board;
  return sidedEval(board, defendersCount - attackersCount / 2);
}
