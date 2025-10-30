export const createQueue = <T>(startSquares: T[]) => {
  const queue = [...startSquares];
  let head = 0;

  return {
    enqueue: (value: T) => {
      queue.push(value);
    },
    dequeue: (): T | undefined => {
      return head < queue.length ? queue[head++] : undefined;
    },
    isEmpty: (): boolean => {
      return head >= queue.length;
    },
  };
};
