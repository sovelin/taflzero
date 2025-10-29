export function createMove(from: number, to: number) {
  return from << 16 | to;
}

export function moveFrom(move: number) {
  return (move >> 16) & 0xFFFF;
}

export function moveTo(move: number) {
  return move & 0xFFFF;
}

export function createMoveFromAlgebraic(move: string) {
  const cols = "abcdefghijk";
  const match = move.match(/^([a-k])([1-9]|10|11)([a-k])([1-9]|10|11)$/);
  if (!match) {
    throw new Error(`Invalid move format: ${move}`);
  }

  const fromCol = cols.indexOf(match[1]);
  const fromRow = parseInt(match[2], 10) - 1;
  const toCol = cols.indexOf(match[3]);
  const toRow = parseInt(match[4], 10) - 1;

  const fromSq = fromRow * 11 + fromCol;
  const toSq = toRow * 11 + toCol;

  return createMove(fromSq, toSq);
}
