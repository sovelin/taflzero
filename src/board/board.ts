import {Piece, Side} from "./types";
import {Board} from "./model/Board";
import {getCol, getRow, getSquareFromAlgebraic} from "./utils";
import {BOARD_SIZE, HOLE, SQS} from "./constants";
import {sideZobrist, zobrist} from "@/board/zobrist/zobrist";

const ATTACKERS_MAX = 24;
const DEFENDERS_MAX = 12;

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
    lastMoveTo: HOLE,
  };
}


function setAttacker(board: Board, sq: number): void {
  if (board.attackersCount >= ATTACKERS_MAX) {
    throw new Error("Exceeded maximum attackers capacity");
  }

  board.attackers[board.attackersCount] = sq;
  board.attackersCount++;
  board.pieceIndexBySquare[sq] = board.attackersCount - 1;
}

function setDefender(board: Board, sq: number): void {
  if (board.defendersCount >= DEFENDERS_MAX) {
    throw new Error("Exceeded maximum defenders capacity");
  }

  board.defenders[board.defendersCount] = sq;
  board.defendersCount++;
  board.pieceIndexBySquare[sq] = board.defendersCount - 1;
}

export function setPiece(board: Board, sq: number, piece: number): void {
  board.board[sq] = piece;
  board.zobrist ^= zobrist[piece - 1][sq];

  const row = getRow(sq);
  const col = getCol(sq);

  board.rowOcc[row] |= (1 << col);
  board.colOcc[col] |= (1 << row);

  if (piece === Piece.ATTACKER) {
    setAttacker(board, sq);
  } else if (piece === Piece.DEFENDER) {
    setDefender(board, sq);
  } else if (piece === Piece.KING) {
    board.kingSq = sq;
  }
}

function clearAttacker(board: Board, sq: number): void {
  const index = board.pieceIndexBySquare[sq];
  if (index !== -1) {
    if (board.attackersCount === 1) {
      board.attackers[index] = HOLE;
      board.pieceIndexBySquare[sq] = 0;
    } else {
      const lastSq = board.attackers[board.attackersCount - 1];
      board.attackers[index] = lastSq;
      board.attackers[board.attackersCount - 1] = HOLE;
      board.pieceIndexBySquare[lastSq] = index;
      board.pieceIndexBySquare[sq] = HOLE;
    }

    board.attackersCount--;
  }
}

function clearDefender(board: Board, sq: number): void {
  const index = board.pieceIndexBySquare[sq];
  if (index !== -1) {
    if (board.defendersCount === 1) {
      board.defenders[index] = HOLE;
      board.pieceIndexBySquare[sq] = 0;
    } else {
      const lastSq = board.defenders[board.defendersCount - 1];
      board.defenders[index] = lastSq;
      board.defenders[board.defendersCount - 1] = HOLE;
      board.pieceIndexBySquare[lastSq] = index;
      board.pieceIndexBySquare[sq] = HOLE;
    }

    board.defendersCount--;
  }
}

export function clearPiece(board: Board, sq: number): void {
  const piece = board.board[sq];
  board.zobrist ^= zobrist[piece - 1][sq];
  board.board[sq] = Piece.EMPTY;

  const row = getRow(sq);
  const col = getCol(sq);

  board.rowOcc[row] &= ~(1 << col);
  board.colOcc[col] &= ~(1 << row);

  if (piece === Piece.ATTACKER) {
    clearAttacker(board, sq);
  } else if (piece === Piece.DEFENDER) {
    clearDefender(board, sq);
  } else if (piece === Piece.KING) {
    board.kingSq = HOLE;
  }
}

export function flipSide(board: Board): void {
  board.sideToMove = board.sideToMove === Side.ATTACKERS ? Side.DEFENDERS : Side.ATTACKERS;
  board.zobrist ^= sideZobrist;
}

export function clearBoard(board: Board): void {
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
  board.lastMoveTo = HOLE;
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

  setAttackers(["d1", "e1", "f1", "g1", "h1"])
  setAttackers(["a4", "a5", "a6", "a7", "a8"])
  setAttackers(["d11", "e11", "f11", "g11", "h11"])
  setAttackers(["k4", "k5", "k6", "k7", "k8"])
  setAttackers(["b6", "f10", "j6", "f2"]);

  setDefenders(["e6", "f5", "f7", "g6"]);
  setDefenders(["d6", "f4", "f8", "h6"]);
  setDefenders(["e5", "e7", "g5", "g7"]);

  setKing("f6");
}
