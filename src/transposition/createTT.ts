
export function createTranspositionTable(sizeMB = 32) {
  const entrySize = 8 + 1 + 2 + 1 + 4;
  const TT_SIZE = (sizeMB * 1024 * 1024) / entrySize | 0;
  const MASK = TT_SIZE - 1;

  const ttZobrist = new BigUint64Array(TT_SIZE);
  const ttDepth   = new Int8Array(TT_SIZE);
  const ttScore   = new Int16Array(TT_SIZE);
  const ttFlag    = new Uint8Array(TT_SIZE);
  const ttMove    = new Uint32Array(TT_SIZE);

  return {
    store(z: bigint, depth: number, score: number, flag: number, move: number) {
      const i = Number(z & BigInt(MASK));
      ttZobrist[i] = z;
      ttDepth[i] = depth;
      ttScore[i] = score;
      ttFlag[i] = flag;
      ttMove[i] = move;
    },
    probe(z: bigint) {
      const i = Number(z & BigInt(MASK));
      if (ttZobrist[i] === z)
        return { depth: ttDepth[i], score: ttScore[i], flag: ttFlag[i], move: ttMove[i] };
      return null;
    },
  };
}
