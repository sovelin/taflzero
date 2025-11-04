// TODO! Constants!

import {MoveGenerator} from "@/moves/movegen/movegen";

const MAX_MOVES = 1024

export const MoveScores = Array.from({length: 256}, () => new Int32Array(MAX_MOVES));

export const estimateMoves = (moveGen: MoveGenerator, moveScores: Int32Array, movesCount: number, ttMove: number | null) => {
  for (let i = 0; i < MAX_MOVES; i++) {
    moveScores[i] = 0;
  }

  for (let i = 0; i < movesCount; i++) {
    moveScores[i] = 0;

    if (moveGen.moves[i] === ttMove) {
      moveScores[i] = 1000000;
    }
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
