import {Board} from "./model/Board";
import {Piece, Side} from "./types";
import {createMove} from "./move";
import {ctz, getBinary, getCol, getRow, getSquare} from "./utils";
import {LINE_MOVES} from "./attackers";
import {BOARD_SIZE} from "./constants";

const THRONE_MASK = 1 << 5;
const BOUNDARY_MASK = 1 | (1 << 10);

const disableBlockedSquares = (beam: number, piece: Piece, rowOrColIndex: number): number => {
  if (piece === Piece.ATTACKER || piece === Piece.DEFENDER) {
    if (rowOrColIndex === 5) {
      beam &= ~THRONE_MASK;
    } else if (rowOrColIndex === 0 || rowOrColIndex === 10) {
      beam &= ~BOUNDARY_MASK;
    }
  }

  return beam;
}

export const createMoveGenerator = () => {
  const MAX_MOVES = 1024;
  const moves = new Uint32Array(MAX_MOVES);
  let moveCount = 0;

  const addMove = (fromSq: number, toSq: number) => {
    if (moveCount >= MAX_MOVES) {
      throw new Error("Exceeded maximum move capacity");
    }

    moves[moveCount++] = createMove(fromSq, toSq);
  }

  const generateHorizontalMoves = (board: Board, fromSq: number) => {
    const row = getRow(fromSq);
    const col = getCol(fromSq);
    const rowOcc = board.rowOcc[row];
    console.log({
      row, col
    })

    for(let i = 0; i < BOARD_SIZE; i++) {
      console.log({
        index: i,
        rowOcc: getBinary(board.rowOcc[i]),
      })
    }

    console.log({
      currentRowOcc: getBinary(rowOcc),
      col,
    })


    let horizontalMoves = LINE_MOVES[col][rowOcc] & ~(1 << col);
    horizontalMoves = disableBlockedSquares(horizontalMoves, board.board[fromSq], row);

    let cur = 1;
    while (horizontalMoves) {
      if (cur & horizontalMoves) {
        const toSq = getSquare(row, ctz(cur));
        addMove(fromSq, toSq);
      }
      horizontalMoves &= ~cur;
      cur <<= 1;
    }
  }

  const generateVerticalMoves = (board: Board, fromSq: number) => {
    const col = getCol(fromSq);
    const row = getRow(fromSq);
    const colOcc = board.colOcc[col];

    let verticalMoves = LINE_MOVES[row][colOcc] & ~(1 << row);
    verticalMoves = disableBlockedSquares(verticalMoves, board.board[fromSq], col);

    let cur = 1;
    while (verticalMoves) {
      if (cur & verticalMoves) {
        const toSq = getSquare(ctz(cur), col)
        addMove(fromSq, toSq);
      }
      verticalMoves &= ~cur;
      cur <<= 1;
    }
  }


  const generateMovesForPiece = (board: Board, fromSq: number) => {
    if (fromSq === -1) return;

    generateHorizontalMoves(board, fromSq);
    generateVerticalMoves(board, fromSq);
  }

  const generateDefenderMoves = (board: Board) => {
    for (let i = 0; i < board.defenders.length; i++) {
      if (board.defenders[i] === -1) continue;
      generateMovesForPiece(board, board.defenders[i]);
    }

    generateMovesForPiece(board, board.kingSq);
  }

  const generateAttackerMoves = (board: Board) => {
    for (let i = 0; i < board.attackers.length; i++) {
      if (board.attackers[i] === -1) continue;
      generateMovesForPiece(board, board.attackers[i]);
    }
  }

  const movegen = (board: Board) => {
    moveCount = 0;

    if (board.sideToMove === Side.ATTACKERS) {
      generateAttackerMoves(board);
    } else {
      generateDefenderMoves(board);
    }
  }

  return {
    movegen,
    moves,
    get movesCount() {
      return moveCount;
    }
  }
}
