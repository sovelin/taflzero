import {search} from "@/search/search";
import {Board} from "@/board";
import {timer} from "@/search/model/Timer";
import {bestMove} from "@/search/model/BestMove";
import {statistics} from "@/search/model/Statistics";

interface Params {
  onIteration?: (depth: number, move: number, score: number, nodes: number, speed: number, time: number) => void;
  time: number;
}

export const searchRoot = function (board: Board, {onIteration, time}: Params) {
  let bestScore = -Infinity;
  let bestMoveRes = 0;
  timer.startSearch(time)
  statistics.reset()

  for (let depth = 1; depth <= 128; depth++) {
    const res = search(board, depth);

    if (!timer.isTimeUp()) {
      bestScore = res;
      bestMoveRes = bestMove.move;
      const speed = statistics.nodes / timer.getTimeElapsed() * 1000;
      onIteration?.(depth, bestMoveRes, bestScore, statistics.nodes, speed, timer.getTimeElapsed());
    }
  }

  return { bestMove: bestMoveRes, bestScore };
}
