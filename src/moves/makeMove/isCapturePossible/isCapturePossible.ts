import {Board, getCornersSq, getSideByPiece, getThroneSq, Piece} from "@/board";

const enemySquares = [...getCornersSq(), getThroneSq()]

const isPotentialThreat = (board: Board,
                           targetSq: number,
                           enemySq: number) => {
  if (board.board[enemySq] === Piece.EMPTY && enemySquares.includes(enemySq)) {
    return true;
  }

  if (board.board[enemySq] === Piece.EMPTY) {
    return false;
  }

  const targetSide = getSideByPiece(board.board[targetSq]);
  const enemySide = getSideByPiece(board.board[enemySq]);

  return targetSide !== enemySide
}

export const isCapturePossible = (
  board: Board,
  targetSq: number,
  enemySq1: number,
  enemySq2: number
) => {
  if (board.board[targetSq] === Piece.KING || board.board[targetSq] === Piece.EMPTY) {
    return false;
  }

  const isThreat1 = isPotentialThreat(board, targetSq, enemySq1);
  const isThreat2 = isPotentialThreat(board, targetSq, enemySq2);

  return isThreat1 && isThreat2;
}
