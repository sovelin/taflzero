import {createQueue} from "@/utils/queue/createQueue";
import {getBottomSquare, getLeftSquare, getRightSquare, getTopSquare} from "@/board/utils";

interface Options {
  isAchievable: (sq: number) => boolean;
  startSquares: number[];
}

export const bfs = (
  {isAchievable, startSquares}: Options
): Set<number> => {
  const queue = createQueue(startSquares);
  const visited = new Set<number>(startSquares);

  while (!queue.isEmpty()) {
    const currentSq = queue.dequeue()!;

    const neighbors = [
      getTopSquare(currentSq),
      getBottomSquare(currentSq),
      getLeftSquare(currentSq),
      getRightSquare(currentSq),
    ].filter((sq) => sq !== null);

    for (const neighbor of neighbors) {
      if (visited.has(neighbor)) continue;
      if (!isAchievable(neighbor)) continue;
      visited.add(neighbor);
      queue.enqueue(neighbor);
    }
  }

  return visited;
}
