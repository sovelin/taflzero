import {BOARD_SIZE} from "@/board";

export function getBinary(num: number, digits = BOARD_SIZE, reverse = true) {
  const masked = num & ((1 << digits) - 1);
  const res = masked.toString(2).padStart(digits, '0');
  return reverse ? res.split('').reverse().join('') : res;
}

export function ctz(x: number) {
  return Math.clz32(x & -x) ^ 31;
}
