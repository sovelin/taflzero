import {createQueue} from "@/utils/queue/createQueue";
import {getBottomSquare, getLeftSquare, getRightSquare, getTopSquare} from "@/board/utils";
import {SQS} from "@/board";

interface Options {
  isAchievable: (sq: number) => boolean;
  startSquares: number[];
}

export const bfs = (
  {isAchievable, startSquares}: Options
) => {
  const queue = createQueue(startSquares);
  //const visited = new Set<number>(startSquares);
  const visitedFlags = new Uint8Array(SQS);

  while (!queue.isEmpty()) {
    const currentSq = queue.dequeue()!;

    const neighbors = [
      getTopSquare(currentSq),
      getBottomSquare(currentSq),
      getLeftSquare(currentSq),
      getRightSquare(currentSq),
    ].filter((sq) => sq !== null);

    for (let i = 0; i < neighbors.length; i++) {
      const neighbor = neighbors[i]!;
      if (visitedFlags[neighbor]) continue;
      if (!isAchievable(neighbor)) continue;
      visitedFlags[neighbor] = 1;
      queue.enqueue(neighbor);
    }
  }

  return visitedFlags;
}
