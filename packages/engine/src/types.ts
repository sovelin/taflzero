export const enum Piece {
  EMPTY = 0,
  ATTACKER = 1,
  DEFENDER = 2,
  KING = 3
}

export const enum Side {
  ATTACKERS = 0,
  DEFENDERS = 1
}

export interface Board {
  // 1) Mailbox (who and where)
  board: Int8Array;            // 121 pieces

  // 2) Pieces list
  attackers: Int16Array;       // 24 attackers indexes or -1 for "holes"
  defenders: Int16Array;       // 12 defenders indexes or -1 for "holes"
  kingSq: number;              // king index or -1 if captured

  attackersCount: number;
  defendersCount: number;

  // 3) fast occupancy access
  rowOcc: Uint16Array;         // 11 rows
  colOcc: Uint16Array;         // 11 columns

  // 4) Params
  sideToMove: Side.ATTACKERS | Side.DEFENDERS;
  moveNumber: number;

  // 5) zobrist and repetition table
  zobrist: bigint;
  repTable: Map<bigint, number>;

  // 6) Rules
  passThroughThrone: boolean; // can the king pass through the throne (Copenhagen rules)?
}
