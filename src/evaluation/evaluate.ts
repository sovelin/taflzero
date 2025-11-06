import {Board, MANHATTAN_DISTANCE, Piece, Side, VERTICAL_HORIZONTAL_NEIGHBORS} from "@/board";
import {KING_SURROUNDING_BONUSES, PieceWeights, PSQT_ATK, PSQT_DEF, PSQT_KING} from "@/evaluation/constants";
import {getPossibleMovesCount} from "./movesCount";

export const sidedEval = (board: Board, score: number): number => {
  return board.sideToMove === Side.DEFENDERS
    ? score
    : -score;
}

const DEFENDERS_MOBILITY_SCORES = [-20, -15, -10, 0, 7, 11, 16, 20, 24, 26, 27, 28, 30, 33, 37, 41, 44, 46, 48, 50, 52];
const ATTACKERS_MOBILITY_SCORES = [-10, -7, -5, 0, 5, 8, 11, 14, 17, 19, 21, 23, 25, 27, 30, 32, 34, 36, 38, 40, 42];
const ATTACKERS_DISTANCE_TO_KING_SCORES = [0, 80, 75, 65, 55, 45, 35, 25, 15, 5, 2,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const DEFENDERS_DISTANCE_TO_KING_SCORES = [0, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 2,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

const evaluateDefendersMobility = (board: Board): number => {
  let res = 0;

  for (let i = 0; i < board.defendersCount; i++) {
    res += DEFENDERS_MOBILITY_SCORES[getPossibleMovesCount(board, board.defenders[i])];
  }

  res += DEFENDERS_MOBILITY_SCORES[getPossibleMovesCount(board, board.kingSq)] * 2;

  return res;
}


const evaluateAttackersMobility = (board: Board): number => {
  let res = 0;

  for (let i = 0; i < board.attackersCount; i++) {
    res += ATTACKERS_MOBILITY_SCORES[getPossibleMovesCount(board, board.attackers[i])];
  }

  return res;
}

const evaluateDistanceToKingAttackerBonuses = (board: Board): number => {
  let res = 0;

  for (let i = 0; i < board.attackersCount; i++) {
    res += ATTACKERS_DISTANCE_TO_KING_SCORES[MANHATTAN_DISTANCE[board.attackers[i]][board.kingSq]];
  }

  return res;
}

const evaluateDistanceToKingDefenderBonuses = (board: Board): number => {
  let res = 0;

  for (let i = 0; i < board.defendersCount; i++) {
    res += DEFENDERS_DISTANCE_TO_KING_SCORES[MANHATTAN_DISTANCE[board.defenders[i]][board.kingSq]];
  }

  return res;
}


export const evaluateBoard = (board: Board): number => {
  const {attackersCount, defendersCount} = board;

  let score = 0
  score += defendersCount * PieceWeights[Piece.DEFENDER];
  score -= attackersCount * PieceWeights[Piece.ATTACKER];

  score += PSQT_KING[board.kingSq];

  score += evaluateDefendersMobility(board);
  score -= evaluateAttackersMobility(board);

  score += evaluateDistanceToKingDefenderBonuses(board);
  score -= evaluateDistanceToKingAttackerBonuses(board);

  for (let i = 0; i < defendersCount; i++) {
    const sq = board.defenders[i];
    score += PSQT_DEF[sq];
  }

  for (let i = 0; i < attackersCount; i++) {
    const sq = board.attackers[i];
    score -= PSQT_ATK[sq];
  }

  let surroundings = 0;

  VERTICAL_HORIZONTAL_NEIGHBORS[board.kingSq]
    .forEach((sq) => {
    if (!sq) return;

    const piece = board.board[sq!]
    if (piece === Piece.ATTACKER) {
      surroundings += 1;
    }
  });


  return sidedEval(board, score + KING_SURROUNDING_BONUSES[surroundings]);
}
