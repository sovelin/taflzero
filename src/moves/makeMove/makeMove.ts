import {
  Board, clearPiece, setPiece, Piece, Side,
  getCol, getCornersSq, getRow, getSideByPiece, getSquare, getThroneSq,
  BOARD_SIZE
} from "@/board";
import {printBoard} from "@/board/print";
import {moveFrom, moveTo} from "../move";

const getSquareByPath = (sq: number, right: number, top: number) => {
  const row = getRow(sq);
  const col = getCol(sq);
  const newRow = row + top;
  const newCol = col + right;

  if (newRow < 0 || newRow >= BOARD_SIZE || newCol < 0 || newCol >= BOARD_SIZE) {
    return -1
  }

  return getSquare(newRow, newCol)
}

const getPotentialPairs = (sq: number) => {
  return [
    getSquareByPath(sq, 2, 0),  // right
    getSquareByPath(sq, -2, 0), // left
    getSquareByPath(sq, 0, 2),  // up
    getSquareByPath(sq, 0, -2), // down
  ].filter((s) => s !== -1);
}

const getBetweenSquare = (sq1: number, sq2: number) => {
  // between square requirements: same row or same column, distance 2
  const row1 = getRow(sq1);
  const col1 = getCol(sq1);
  const row2 = getRow(sq2);
  const col2 = getCol(sq2);

  // same row
  if (row1 === row2 && Math.abs(col1 - col2) === 2) {
    return getSquare(row1, (col1 + col2) / 2);
  }

  // same column
  if (col1 === col2 && Math.abs(row1 - row2) === 2) {
    return getSquare((row1 + row2) / 2, col1);
  }

  // not valid
  return -1;
}

const enemySquares = [...getCornersSq(), getThroneSq()]

// TODO: Separate to module + tests
const isCapturePossible = (
  board: Board,
  targetSq: number,
  enemySq1: number,
  enemySq2: number
) => {
  const targetPiece = board.board[targetSq];
  const enemyPiece1 = board.board[enemySq1];
  const enemyPiece2 = board.board[enemySq2];

  if (!enemyPiece1 || !enemyPiece2 || !targetPiece) {
    return false;
  }

  if (targetPiece === Piece.KING) {
    return false;
  }

  const enemySq1Side = getSideByPiece(enemyPiece1);
  const enemySq2Side = getSideByPiece(enemyPiece2);
  const targetSide = getSideByPiece(targetPiece);

  // Both enemy squares must belong to the same side, which is different from the target side
  if (enemySq1Side === enemySq2Side && enemySq1Side !== targetSide && enemySq1Side !== null) {
    return true;
  }

  // At least one square is same side -> no capture
  if (enemySq1Side === targetSide || enemySq2Side === targetSide) {
    return false;
  }

  // One of the enemy squares is special square
  if (enemySquares.includes(enemySq1)) {
    return enemySq2Side !== null && enemySq2Side !== targetSide;
  }
  if (enemySquares.includes(enemySq2)) {
    return enemySq1Side !== null && enemySq1Side !== targetSide;
  }

  return false;
}

export const makeMove = (board: Board, move: number) => {
  const fromSq = moveFrom(move)
  const toSq = moveTo(move)
  const piece = board.board[fromSq];

  clearPiece(board, fromSq);
  setPiece(board, toSq, piece);

  // Captures check
  const potentialPairs = getPotentialPairs(toSq);
  printBoard(board)
  for (const pairSq of potentialPairs) {
    const betweenSq = getBetweenSquare(toSq, pairSq);

    if (betweenSq === -1) {
      continue;
    }

    // Check if between piece is opponent and pair piece is ally
    if (isCapturePossible(board, betweenSq, toSq, pairSq)) {
      clearPiece(board, betweenSq);
    }
  }

  board.sideToMove = board.sideToMove === Side.ATTACKERS ? Side.DEFENDERS : Side.ATTACKERS;
}
