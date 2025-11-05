import {BOARD_SIZE} from "@/board";

export const LINE_MOVES: Uint16Array[] =
  Array.from({ length: BOARD_SIZE }, () => new Uint16Array(1 << BOARD_SIZE));

export const POSSIBLE_MOVES_COUNT : Uint8Array[] =
  Array.from({ length: BOARD_SIZE }, () => new Uint8Array(1 << BOARD_SIZE));


function buildLineMask(at: number, occ: number): number {
  let mask = 0;
  for (let i = at - 1; i >= 0; i--) {
    if (occ & (1 << i)) break;
    mask |= (1 << i);
  }
  for (let i = at + 1; i < BOARD_SIZE; i++) {
    if (occ & (1 << i)) break;
    mask |= (1 << i);
  }
  return mask;
}
function countSetBits(n: number): number {
  let count = 0;
  while (n) {
    count += n & 1;
    n >>= 1;
  }
  console.log({count})
  return count;
}

export function initLineMoves() {
  for (let pos = 0; pos < BOARD_SIZE; pos++) {
    for (let occ = 0; occ < (1 << BOARD_SIZE); occ++) {
      LINE_MOVES[pos][occ] = buildLineMask(pos, occ);
      POSSIBLE_MOVES_COUNT[pos][occ] = countSetBits(LINE_MOVES[pos][occ]);
    }
  }
}

export function printLineMaskExamples(pos: number, occ: number) {
  console.log(`Line moves example for position ${pos} with occupancy ${occ.toString(2).padStart(BOARD_SIZE, '0')}:`);
  const mask = LINE_MOVES[pos][occ];
  console.log(mask.toString(2).padStart(BOARD_SIZE, '0'));
}
