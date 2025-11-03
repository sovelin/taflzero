import {Board, BOARD_SIZE} from "@/board";
import {makeMove} from "@/moves";

export function getBinary(num: number, digits = BOARD_SIZE, reverse = true) {
  const masked = num & ((1 << digits) - 1);
  const res = masked.toString(2).padStart(digits, '0');
  return reverse ? res.split('').reverse().join('') : res;
}

export function ctz(x: number) {
  return Math.clz32(x & -x) ^ 31;
}

export function setMoves(board: Board, moves: number[]) {
  moves.forEach(move => {
    makeMove(board, move);
  })
}
