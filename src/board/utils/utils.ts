import {BOARD_SIZE} from "../constants";
import {Piece, Side} from "../types";

export function getSquare(row: number, col: number): number {
  return row * BOARD_SIZE + col;
}

export function getSquareFromAlgebraic(coord: string): number {
  const file = coord.charCodeAt(0) - "a".charCodeAt(0);
  const rank = parseInt(coord.slice(1), 10) - 1;
  return getSquare(rank, file);
}

export function getRow(sq: number): number {
  return Math.floor(sq / BOARD_SIZE);
}

export function getCol(sq: number): number {
  return sq % BOARD_SIZE;
}

export function getSideByPiece(piece: number) {
  switch (piece) {
    case Piece.ATTACKER:
      return Side.ATTACKERS;
    case Piece.DEFENDER:
    case Piece.KING:
      return Side.DEFENDERS;
    default:
      return null;
  }
}

export const getThroneSq = () => getSquare(
  5,
  5
)

export const getCornersSq = () =>[
  getSquare(
    0,
   0
  ),
  getSquare(
   0,
   BOARD_SIZE - 1
  ),
  getSquare(
    BOARD_SIZE - 1,
    0
  ),
  getSquare(
    BOARD_SIZE - 1,
    BOARD_SIZE - 1
  ),
]

export function getSquareAlg(sq: number) {
  const cols = "abcdefghijk";
  const row = getRow(sq)
  const col = getCol(sq)
  return `${cols[col]}${row + 1}`;
}

export function getOppositeSide(side: Side): Side {
  return side === Side.ATTACKERS ? Side.DEFENDERS : Side.ATTACKERS;
}

export function getTopLeftSquare(): number {
  return getSquare(BOARD_SIZE - 1, 0);
}

export function getBottomRightSquare(): number {
  return getSquare(0, BOARD_SIZE - 1);
}

export function getTopRightSquare(): number {
  return getSquare(BOARD_SIZE - 1, BOARD_SIZE - 1);
}

export function getBottomLeftSquare(): number {
  return getSquare(0, 0);
}
