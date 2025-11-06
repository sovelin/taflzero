import {SQS} from "@/board";
import {MAX_HISTORY_SCORE} from "@/search/constants";

export const history = Array.from({ length: 2 }, () =>
  Array.from({ length: SQS }, () =>
    new Int32Array(SQS)
  ));

export const updateHistory = (side: number, from: number, to: number, depth: number) => {
  history[side][from][to] += depth * depth;

  if (history[side][from][to] > MAX_HISTORY_SCORE) {
    history[side][from][to] >>= 1;
  }
}
