import {
  Board, clearPiece, setPiece,
  getCol, getRow, getSquare,
  BOARD_SIZE,
} from "@/board";
import {moveFrom, moveTo} from "../move";
import {CapturedPiece, UndoMove} from "../model/UndoMove";
import {isCapturePossible} from "./isCapturePossible";
import {makeShieldWallCaptures} from "./makeShieldCaptures";
import {flipSide} from "@/board/board";

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

const addPositionToRepTable = (board: Board) => {
  if (board.repTable.has(board.zobrist)) {
    board.repTable.set(board.zobrist, board.repTable.get(board.zobrist)! + 1);
  } else {
    board.repTable.set(board.zobrist, 1);
  }
}


export const makeMove = (board: Board, move: number): UndoMove => {
  const fromSq = moveFrom(move)
  const toSq = moveTo(move)
  const piece = board.board[fromSq];
  const lastMoveTo = board.lastMoveTo

  clearPiece(board, fromSq);
  setPiece(board, toSq, piece);

  // Captures check
  const potentialPairs = getPotentialPairs(toSq);
  const captured: CapturedPiece[] = [];

  for (const pairSq of potentialPairs) {
    const betweenSq = getBetweenSquare(toSq, pairSq);

    if (betweenSq === -1) {
      continue;
    }

    // Check if between piece is opponent and pair piece is ally
    if (isCapturePossible(board, betweenSq, toSq, pairSq)) {
      captured.push({
        sq: betweenSq,
        piece: board.board[betweenSq],
      })
      clearPiece(board, betweenSq);
    }
  }

  const shieldWallCaptures = makeShieldWallCaptures(board, toSq, board.sideToMove);
  flipSide(board);
  board.lastMoveTo = toSq;

  addPositionToRepTable(board);

  return {
    from: fromSq,
    to: toSq,
    captured: shieldWallCaptures.length > 0 ? [...captured, ...shieldWallCaptures] : captured,
    movedPiece: piece,
    lastMoveTo,
  }
}
