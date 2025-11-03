import {Board, Piece, Side} from "@/board";
import {PieceWeights, PSQT_ATK, PSQT_DEF, PSQT_KING} from "@/evaluation/constants";

export const sidedEval = (board: Board, score: number): number => {
  return board.sideToMove === Side.DEFENDERS
    ? score
    : -score;
}

export const evaluateBoard = (board: Board): number => {
  // Simple evaluation function: difference in number of pieces
  const {attackersCount, defendersCount} = board;

  let score = 0
  score += defendersCount * PieceWeights[Piece.DEFENDER];
  score -= attackersCount * PieceWeights[Piece.ATTACKER];

  score += PSQT_KING[board.kingSq];

  for (let i = 0; i < defendersCount; i++) {
    const sq = board.defenders[i];
    score += PSQT_DEF[sq];
  }

  for (let i = 0; i < attackersCount; i++) {
    const sq = board.attackers[i];
    score -= PSQT_ATK[sq];
  }

  return sidedEval(board, score);
}
