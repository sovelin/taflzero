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

export function getRightNeighbor(sq: number): number | null {
  const col = getCol(sq);
  if (col === BOARD_SIZE - 1) return null;
  return sq + 1;
}

export function getLeftNeighbor(sq: number): number | null {
  const col = getCol(sq);
  if (col === 0) return null;
  return sq - 1;
}

export function getTopNeighbor(sq: number): number | null {
  const row = getRow(sq);
  if (row === BOARD_SIZE - 1) return null;
  return sq + BOARD_SIZE;
}

export function getBottomNeighbor(sq: number): number | null {
  const row = getRow(sq);
  if (row === 0) return null;
  return sq - BOARD_SIZE;
}

export function getTopLeftNeighbor(sq: number): number | null {
  const row = getRow(sq);
  const col = getCol(sq);
  if (row === BOARD_SIZE - 1 || col === 0) return null;
  return sq + BOARD_SIZE - 1;
}

export function getTopRightNeighbor(sq: number): number | null {
  const row = getRow(sq);
  const col = getCol(sq);
  if (row === BOARD_SIZE - 1 || col === BOARD_SIZE - 1) return null;
  return sq + BOARD_SIZE + 1;
}

export function getBottomLeftNeighbor(sq: number): number | null {
  const row = getRow(sq);
  const col = getCol(sq);
  if (row === 0 || col === 0) return null;
  return sq - BOARD_SIZE - 1;
}

export function getBottomRightNeighbor(sq: number): number | null {
  const row = getRow(sq);
  const col = getCol(sq);
  if (row === 0 || col === BOARD_SIZE - 1) return null;
  return sq - BOARD_SIZE + 1;
}

export function getVerticalHorizontalNeighbors(sq: number): number[] {
  const siblings: number[] = [];
  const top = getTopNeighbor(sq);
  const bottom = getBottomNeighbor(sq);
  const left = getLeftNeighbor(sq);
  const right = getRightNeighbor(sq);

  if (top !== null) siblings.push(top);
  if (bottom !== null) siblings.push(bottom);
  if (left !== null) siblings.push(left);
  if (right !== null) siblings.push(right);

  return siblings;
}

export function getAllNeighbors(sq: number): number[] {
  const siblings: number[] = [];
  const verticalHorizontal = getVerticalHorizontalNeighbors(sq);
  siblings.push(...verticalHorizontal);

  const topLeft = getTopLeftNeighbor(sq);
  const topRight = getTopRightNeighbor(sq);
  const bottomLeft = getBottomLeftNeighbor(sq);
  const bottomRight = getBottomRightNeighbor(sq);

  if (topLeft !== null) siblings.push(topLeft);
  if (topRight !== null) siblings.push(topRight);
  if (bottomLeft !== null) siblings.push(bottomLeft);
  if (bottomRight !== null) siblings.push(bottomRight);

  return siblings;
}

export function isEdgeSquare(sq: number): boolean {
  const row = getRow(sq);
  const col = getCol(sq);
  return row === 0 || row === BOARD_SIZE - 1 || col === 0 || col === BOARD_SIZE - 1;
}
