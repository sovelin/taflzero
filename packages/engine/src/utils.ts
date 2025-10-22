import {BOARD_SIZE} from "./constants";

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

export function getBinary(num: number, digits: number) {
  return num.toString(2).padStart(digits, '0');
}
