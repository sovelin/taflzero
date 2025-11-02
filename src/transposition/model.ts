const enum TTFlag {
  EXACT = 0,
  LOWERBOUND = 1,
  UPPERBOUND = 2
}

interface TTEntry {
  zobrist: bigint;
  depth: number;
  score: number;
  flag: TTFlag;
  bestMove: number;
}

function initTT() {
  const TT_SIZE = 1 << 20; // например, 1M записей
  const TT_MASK = TT_SIZE - 1;
}
