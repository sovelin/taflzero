import {createQueue} from "@/utils/queue/createQueue";
import {
  getVerticalHorizontalNeighbors
} from "@/board/utils";
import {SQS} from "@/board";

interface Options {
  isAchievable: (sq: number) => boolean;
  startSquares: number[];
  getNeighbors?: (sq: number) => number[];
}

export const bfs = (
  {isAchievable, startSquares, getNeighbors = getVerticalHorizontalNeighbors}: Options
) => {
  const queue = createQueue(startSquares);
  const visitedFlags = new Uint8Array(SQS);

  while (!queue.isEmpty()) {
    const currentSq = queue.dequeue()!;

    const neighbors = getNeighbors(currentSq)

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
