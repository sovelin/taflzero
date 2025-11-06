// TODO! Constants!

import {MoveGenerator} from "@/moves/movegen/movegen";
import {killers} from "@/search/model/Killers";
import {moveFrom, moveTo} from "@/moves";
import {Side} from "@/board";
import {history} from "@/search/model/History";

const MAX_MOVES = 1024

export const MoveScores = Array.from({length: 256}, () => new Int32Array(MAX_MOVES));

const clampHistoryScore = (score: number) => {
  return score / 1000; // scale down to prevent overflow (will be in range [0, 1000])
}

export const estimateMoves = (moveGen: MoveGenerator, moveScores: Int32Array, movesCount: number, ttMove: number | null, height: number, side: Side) => {
  for (let i = 0; i < MAX_MOVES; i++) {
    moveScores[i] = 0;
  }

  for (let i = 0; i < movesCount; i++) {
    moveScores[i] = 0;

    if (moveGen.moves[i] === ttMove) {
      moveScores[i] += 1000000;
    }

    if (moveGen.moves[i] === killers[height][0]) {
      moveScores[i] += 900000;
    } else if (moveGen.moves[i] === killers[height][1]) {
      moveScores[i] += 800000;
    }

    // History heuristic: from^to
    const from = moveFrom(moveGen.moves[i])
    const to = moveTo(moveGen.moves[i])
    const historyScore = history[side][from][to];
    moveScores[i] += clampHistoryScore(historyScore);
  }
}

export const pickMove = (moveGen: MoveGenerator, moveScores: Int32Array) => {
  if (moveGen.movesCount === 0) {
    return 0;
  }

  let bestIndex = 0;
  let bestScore = moveScores[0];

  for (let i = 1; i < moveGen.movesCount; i++) {
    if (moveScores[i] > bestScore) {
      bestScore = moveScores[i];
      bestIndex = i;
    }
  }

  const bestMove = moveGen.moves[bestIndex];

  // remove by swapping last and best, length--
  moveGen.moves[bestIndex] = moveGen.moves[moveGen.movesCount - 1];
  moveScores[bestIndex] = moveScores[moveGen.movesCount - 1];

  moveGen.decreaseCount()

  return bestMove;
}
