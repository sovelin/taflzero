import {BOARD_SIZE} from "@/board";

export const printBoardSet = (set: Set<number>) => {
  for (let row = BOARD_SIZE; row >= 0; row--) {
    let rowStr = '';
    for (let col = 0; col < BOARD_SIZE; col++) {
      const sq = row * BOARD_SIZE + col;
      if (set.has(sq)) {
        rowStr += ' 1 ';
      } else {
        rowStr += ' . ';
      }
    }
    console.log(rowStr);
  }
}
