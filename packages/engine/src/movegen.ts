import {Board} from "./model/Board";
import {Side} from "./types";
import {createMove} from "./move";
import {getBinary, getCol, getRow, getSquare} from "./utils";
import {LINE_MOVES} from "./attackers";
import {BOARD_SIZE} from "./constants";

const MAX_MOVES = 1024;
export const moves = new Uint32Array(MAX_MOVES);
let moveCount = 0;
export const getMovesCount = () => moveCount;

function addMove(fromSq: number, toSq: number) {
  if (moveCount >= MAX_MOVES) {
    throw new Error("Exceeded maximum move capacity");
  }

  moves[moveCount++] = createMove(fromSq, toSq);
}

function generateHorizontalMoves(board: Board, fromSq: number) {
  const row = getRow(fromSq);
  const rowOcc = board.rowOcc[row];
  let horizontalMoves = LINE_MOVES[row][rowOcc];
  //printLineMaskExamples(rowOcc, row);
  // TODO: check disable fields

  let cur = 1;
  while(horizontalMoves) {
    console.log({
      map: getBinary(horizontalMoves, BOARD_SIZE),
      index: getBinary(cur, BOARD_SIZE),
      combined: getBinary(cur & horizontalMoves, BOARD_SIZE),
    });
    if (cur & horizontalMoves) {
      const toSq = getSquare(row, cur)
      addMove(fromSq, toSq);
    }
    horizontalMoves &= ~cur;
    cur <<= 1;
  }
}

function generateVerticalMoves(board: Board, fromSq: number) {
  const col = getCol(fromSq);
  const colOcc = board.colOcc[col];
  let verticalMoves = LINE_MOVES[col][colOcc];
  // TODO: check disable fields

  let cur = 1;
  while (verticalMoves) {
    if (cur & verticalMoves) {
      const toSq = getSquare(cur, col)
      addMove(fromSq, toSq);
    }
    verticalMoves &= ~cur;
    cur <<= 1;
  }
}


function generateMovesForPiece(board: Board, fromSq: number) {
  generateHorizontalMoves(board, fromSq);
  generateVerticalMoves(board, fromSq);
}

function generateDefenderMoves(board: Board) {
  for (let i = 0; i < board.defenders.length; i++) {
    if (board.defenders[i] === -1) continue;
    generateMovesForPiece(board, board.defenders[i]);
  }

  generateMovesForPiece(board, board.kingSq);
}

function generateAttackerMoves(board: Board) {
  for (let i = 0; i < board.attackers.length; i++) {
    if (board.attackers[i] === -1) continue;
    generateMovesForPiece(board, board.attackers[i]);
  }
}

export function movegen(board: Board) {
  moveCount = 0;

  if (board.sideToMove === Side.ATTACKERS) {
    generateAttackerMoves(board);
  } else {
    generateDefenderMoves(board);
  }
}
