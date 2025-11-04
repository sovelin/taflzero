import {createQueue} from "@/utils/queue/createQueue";
import {SQS, VERTICAL_HORIZONTAL_NEIGHBORS} from "@/board";

interface Options {
  isAchievable: (sq: number) => boolean;
  startSquares: number[];
  neighbors?: number[][];
}

export const bfs = (
  {isAchievable, startSquares, neighbors: neighborsCached = VERTICAL_HORIZONTAL_NEIGHBORS}: Options
) => {
  const queue = createQueue(startSquares);
  const visitedFlags = new Uint8Array(SQS);

  while (!queue.isEmpty()) {
    const currentSq = queue.dequeue()!;
    const neighbors = neighborsCached[currentSq]

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
