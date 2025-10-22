import {Piece, Side} from "./types";
import {Board} from "./model/Board";
import {getCol, getRow, getSquareFromAlgebraic} from "./utils";
import {BOARD_SIZE, SQS} from "./constants";

const ATTACKERS_MAX = 24;
const DEFENDERS_MAX = 12;
const HOLE = -1;

export function createBoard(): Board {
  return {
    board: new Int8Array(SQS).fill(0),
    attackers: new Int16Array(ATTACKERS_MAX).fill(HOLE),
    defenders: new Int16Array(DEFENDERS_MAX).fill(HOLE),
    kingSq: HOLE,

    attackersCount: 0,
    defendersCount: 0,

    rowOcc: new Uint16Array(BOARD_SIZE),
    colOcc: new Uint16Array(BOARD_SIZE),
    pieceIndexBySquare: new Int16Array(SQS).fill(HOLE),

    sideToMove: Side.ATTACKERS,
    moveNumber: 1,

    zobrist: 0n,
    repTable: new Map(),

    passThroughThrone: true,
  };
}

export function setPiece(board: Board, sq: number, piece: number): void {
  board.board[sq] = piece;

  const row = getRow(sq);
  const col = getCol(sq);

  board.rowOcc[row] |= (1 << row);
  board.colOcc[col] |= (1 << col);

  if (piece === Piece.ATTACKER) {
    board.attackers[board.attackersCount] = sq;
    board.attackersCount++;
  } else if (piece === Piece.DEFENDER) {
    board.defenders[board.defendersCount] = sq;
    board.defendersCount++;
  } else if (piece === Piece.KING) {
    board.kingSq = sq;
  }
}

export function clearPiece(board: Board, sq: number): void {
  const piece = board.board[sq];
  board.board[sq] = Piece.EMPTY;

  const row = getRow(sq);
  const col = getCol(sq);

  board.rowOcc[row] &= ~(1 << row);
  board.colOcc[col] &= ~(1 << col);

  if (piece === Piece.ATTACKER) {
    const index = board.attackers.findIndex(s => s === sq);
    if (index !== -1) {
      board.attackers[index] = HOLE;
      board.attackersCount--;
    }
  } else if (piece === Piece.DEFENDER) {
    const index = board.defenders.findIndex(s => s === sq);
    if (index !== -1) {
      board.defenders[index] = HOLE;
      board.defendersCount--;
    }
  } else if (piece === Piece.KING) {
    board.kingSq = HOLE;
  }
}

function clearBoard(board: Board): void {
  board.board.fill(Piece.EMPTY);
  board.attackers.fill(HOLE);
  board.defenders.fill(HOLE);
  board.kingSq = HOLE;
  board.attackersCount = 0;
  board.defendersCount = 0;
  board.rowOcc.fill(0);
  board.colOcc.fill(0);
  board.pieceIndexBySquare.fill(HOLE);
  board.sideToMove = Side.ATTACKERS;
  board.moveNumber = 1;
  board.zobrist = 0n;
  board.repTable.clear();
}

export function setInitialPosition(board: Board): void {
  clearBoard(board);

  const setPieces = (positions: string[], piece: Piece) => {
    for (const pos of positions) {
      const sq = getSquareFromAlgebraic(pos);
      setPiece(board, sq, piece);
    }
  }

  const setAttackers = (positions: string[]) => {
    setPieces(positions, Piece.ATTACKER);
  }

  const setDefenders = (positions: string[]) => {
    setPieces(positions, Piece.DEFENDER);
  }

  const setKing = (position: string) => {
    const sq = getSquareFromAlgebraic(position);
    setPiece(board, sq, Piece.KING);
  }

  setAttackers(["d1", "e1", "f1", "g1", "h1",])
  setAttackers(["a4", "a5", "a6", "a7", "a8",])
  setAttackers(["d11", "e11", "f11", "g11", "h11",])
  setAttackers(["k4", "k5", "k6", "k7", "k8",])
  setAttackers(["b6", "f10", "j6", "f2"]);

  setDefenders(["e6", "f5", "f7", "g6"]);
  setDefenders(["d6", "f4", "f8", "h6"]);
  setDefenders(["e5", "e7", "g5", "g7"]);

  setKing("f6");
}
