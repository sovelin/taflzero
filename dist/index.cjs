"use strict";
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/lib.ts
var lib_exports = {};
__export(lib_exports, {
  BOARD_SIZE: () => BOARD_SIZE,
  HOLE: () => HOLE,
  INITIAL_FEN: () => INITIAL_FEN,
  MATE_SCORE: () => MATE_SCORE,
  NUM_PIECE_KINDS: () => NUM_PIECE_KINDS,
  Piece: () => Piece,
  SQS: () => SQS,
  Side: () => Side,
  bestMove: () => bestMove,
  bfs: () => bfs,
  checkTerminal: () => checkTerminal,
  clearBoard: () => clearBoard,
  clearPiece: () => clearPiece,
  createBoard: () => createBoard,
  createMove: () => createMove,
  createMoveFromAlgebraic: () => createMoveFromAlgebraic,
  createMoveGenerator: () => createMoveGenerator,
  ctz: () => ctz,
  evaluateBoard: () => evaluateBoard,
  getBinary: () => getBinary,
  getBottomLeftSquare: () => getBottomLeftSquare,
  getBottomRightSquare: () => getBottomRightSquare,
  getCol: () => getCol,
  getCornersSq: () => getCornersSq,
  getFEN: () => getFEN,
  getMoveAlg: () => getMoveAlg,
  getOppositeSide: () => getOppositeSide,
  getRow: () => getRow,
  getScoreText: () => getScoreText,
  getSideByPiece: () => getSideByPiece,
  getSquare: () => getSquare,
  getSquareAlg: () => getSquareAlg,
  getSquareFromAlgebraic: () => getSquareFromAlgebraic,
  getThroneSq: () => getThroneSq,
  getTopLeftSquare: () => getTopLeftSquare,
  getTopRightSquare: () => getTopRightSquare,
  initMovesModule: () => initMovesModule,
  isMateScore: () => isMateScore,
  makeMove: () => makeMove,
  moveFrom: () => moveFrom,
  moveTo: () => moveTo,
  search: () => search,
  setFEN: () => setFEN,
  setInitialPosition: () => setInitialPosition,
  setPiece: () => setPiece,
  sidedEval: () => sidedEval,
  statistics: () => statistics,
  unmakeMove: () => unmakeMove
});
module.exports = __toCommonJS(lib_exports);

// src/search/model/Statistics.ts
var Statistics = class {
  nodes = 0;
  reset() {
    this.nodes = 0;
  }
  incrementNodes(count = 1) {
    this.nodes += count;
  }
};
var statistics = new Statistics();

// src/board/types.ts
var Piece = /* @__PURE__ */ ((Piece2) => {
  Piece2[Piece2["EMPTY"] = 0] = "EMPTY";
  Piece2[Piece2["ATTACKER"] = 1] = "ATTACKER";
  Piece2[Piece2["DEFENDER"] = 2] = "DEFENDER";
  Piece2[Piece2["KING"] = 3] = "KING";
  return Piece2;
})(Piece || {});
var Side = /* @__PURE__ */ ((Side3) => {
  Side3[Side3["ATTACKERS"] = 0] = "ATTACKERS";
  Side3[Side3["DEFENDERS"] = 1] = "DEFENDERS";
  return Side3;
})(Side || {});

// src/board/constants.ts
var BOARD_SIZE = 11;
var SQS = BOARD_SIZE * BOARD_SIZE;
var HOLE = -1;
var NUM_PIECE_KINDS = 3;
var INITIAL_FEN = "3aaaaa3/5a5/11/a4d4a/a3ddd3a/aa1ddkdd1aa/a3ddd3a/a4d4a/11/5a5/3aaaaa3 a";

// src/board/utils/utils.ts
function getSquare(row, col) {
  return row * BOARD_SIZE + col;
}
function getSquareFromAlgebraic(coord) {
  const file = coord.charCodeAt(0) - "a".charCodeAt(0);
  const rank = parseInt(coord.slice(1), 10) - 1;
  return getSquare(rank, file);
}
function getRow(sq) {
  return Math.floor(sq / BOARD_SIZE);
}
function getCol(sq) {
  return sq % BOARD_SIZE;
}
function getSideByPiece(piece) {
  switch (piece) {
    case 1 /* ATTACKER */:
      return 0 /* ATTACKERS */;
    case 2 /* DEFENDER */:
    case 3 /* KING */:
      return 1 /* DEFENDERS */;
    default:
      return null;
  }
}
var getThroneSq = () => getSquare(
  5,
  5
);
var getCornersSq = () => [
  getSquare(
    0,
    0
  ),
  getSquare(
    0,
    BOARD_SIZE - 1
  ),
  getSquare(
    BOARD_SIZE - 1,
    0
  ),
  getSquare(
    BOARD_SIZE - 1,
    BOARD_SIZE - 1
  )
];
function getSquareAlg(sq) {
  const cols = "abcdefghijk";
  const row = getRow(sq);
  const col = getCol(sq);
  return `${cols[col]}${row + 1}`;
}
function getOppositeSide(side) {
  return side === 0 /* ATTACKERS */ ? 1 /* DEFENDERS */ : 0 /* ATTACKERS */;
}
function getTopLeftSquare() {
  return getSquare(BOARD_SIZE - 1, 0);
}
function getBottomRightSquare() {
  return getSquare(0, BOARD_SIZE - 1);
}
function getTopRightSquare() {
  return getSquare(BOARD_SIZE - 1, BOARD_SIZE - 1);
}
function getBottomLeftSquare() {
  return getSquare(0, 0);
}
function getRightNeighbor(sq) {
  const col = getCol(sq);
  if (col === BOARD_SIZE - 1) return null;
  return sq + 1;
}
function getLeftNeighbor(sq) {
  const col = getCol(sq);
  if (col === 0) return null;
  return sq - 1;
}
function getTopNeighbor(sq) {
  const row = getRow(sq);
  if (row === BOARD_SIZE - 1) return null;
  return sq + BOARD_SIZE;
}
function getBottomNeighbor(sq) {
  const row = getRow(sq);
  if (row === 0) return null;
  return sq - BOARD_SIZE;
}
function getTopLeftNeighbor(sq) {
  const row = getRow(sq);
  const col = getCol(sq);
  if (row === BOARD_SIZE - 1 || col === 0) return null;
  return sq + BOARD_SIZE - 1;
}
function getTopRightNeighbor(sq) {
  const row = getRow(sq);
  const col = getCol(sq);
  if (row === BOARD_SIZE - 1 || col === BOARD_SIZE - 1) return null;
  return sq + BOARD_SIZE + 1;
}
function getBottomLeftNeighbor(sq) {
  const row = getRow(sq);
  const col = getCol(sq);
  if (row === 0 || col === 0) return null;
  return sq - BOARD_SIZE - 1;
}
function getBottomRightNeighbor(sq) {
  const row = getRow(sq);
  const col = getCol(sq);
  if (row === 0 || col === BOARD_SIZE - 1) return null;
  return sq - BOARD_SIZE + 1;
}
function getVerticalHorizontalNeighbors(sq) {
  const siblings = [];
  const top = getTopNeighbor(sq);
  const bottom = getBottomNeighbor(sq);
  const left = getLeftNeighbor(sq);
  const right = getRightNeighbor(sq);
  if (top !== null) siblings.push(top);
  if (bottom !== null) siblings.push(bottom);
  if (left !== null) siblings.push(left);
  if (right !== null) siblings.push(right);
  return siblings;
}
function getAllNeighbors(sq) {
  const siblings = [];
  const verticalHorizontal = getVerticalHorizontalNeighbors(sq);
  siblings.push(...verticalHorizontal);
  const topLeft = getTopLeftNeighbor(sq);
  const topRight = getTopRightNeighbor(sq);
  const bottomLeft = getBottomLeftNeighbor(sq);
  const bottomRight = getBottomRightNeighbor(sq);
  if (topLeft !== null) siblings.push(topLeft);
  if (topRight !== null) siblings.push(topRight);
  if (bottomLeft !== null) siblings.push(bottomLeft);
  if (bottomRight !== null) siblings.push(bottomRight);
  return siblings;
}
function isEdgeSquare(sq) {
  const row = getRow(sq);
  const col = getCol(sq);
  return row === 0 || row === BOARD_SIZE - 1 || col === 0 || col === BOARD_SIZE - 1;
}

// src/utils/random.ts
var rngState = 0x9e3779b97f4a7c15n;
function random64() {
  let x = rngState;
  x ^= x >> 12n;
  x ^= x << 25n;
  x ^= x >> 27n;
  rngState = x;
  return x * 0x2545F4914F6CDD1Dn & 0xFFFFFFFFFFFFFFFFn;
}

// src/board/zobrist/zobrist.ts
var zobrist = Array.from(
  { length: NUM_PIECE_KINDS },
  () => Array.from({ length: SQS }, () => random64())
);
var sideZobrist = random64();

// src/board/fen/fen.ts
var isValidFENChar = (char) => {
  return "kad0123456789".includes(char);
};
var charToPiece = {
  "k": 3 /* KING */,
  "a": 1 /* ATTACKER */,
  "d": 2 /* DEFENDER */
};
var charToSide = {
  "a": 0 /* ATTACKERS */,
  "d": 1 /* DEFENDERS */
};
var pieceToChar = {
  [3 /* KING */]: "k",
  [1 /* ATTACKER */]: "a",
  [2 /* DEFENDER */]: "d"
};
var sideToChar = {
  [0 /* ATTACKERS */]: "a",
  [1 /* DEFENDERS */]: "d"
};
var isNumberChar = (char) => {
  return "0123456789".includes(char);
};
var setRowFromFEN = (rowFEN, rowIndex, board) => {
  let colIndex = 0;
  let numberCache = "";
  for (let i = 0; i < rowFEN.length; i++) {
    const char = rowFEN[i];
    if (!isValidFENChar(char)) {
      throw new Error(`Invalid FEN character: ${char}`);
    }
    if (isNumberChar(char)) {
      numberCache += char;
      continue;
    }
    if (numberCache.length > 0) {
      const number = parseInt(numberCache, 10);
      colIndex += number;
      numberCache = "";
    }
    const sq = getSquare(rowIndex, colIndex);
    setPiece(board, sq, charToPiece[char]);
    colIndex++;
  }
};
var setFEN = (board, fen) => {
  clearBoard(board);
  const [rows, side] = fen.split(" ");
  const splittedRows = rows.split("/");
  if (splittedRows.length !== BOARD_SIZE) {
    throw new Error(`Invalid FEN rows: ${rows}`);
  }
  splittedRows.forEach((rowFEN, rowIndex) => {
    setRowFromFEN(rowFEN, BOARD_SIZE - 1 - rowIndex, board);
  });
  const sideToSet = charToSide[side];
  if (board.sideToMove !== sideToSet) {
    flipSide(board);
  }
};
var getFEN = (board) => {
  let fenRows = [];
  for (let row = BOARD_SIZE - 1; row >= 0; row--) {
    let emptyCount = 0;
    let fenRow = "";
    for (let col = 0; col < BOARD_SIZE; col++) {
      const sq = getSquare(row, col);
      const piece = board.board[sq];
      if (piece === 0 /* EMPTY */) {
        emptyCount++;
        continue;
      }
      if (emptyCount > 0) {
        fenRow += emptyCount.toString();
        emptyCount = 0;
      }
      fenRow += pieceToChar[piece];
    }
    if (emptyCount > 0) {
      fenRow += emptyCount.toString();
    }
    fenRows.push(fenRow);
  }
  return fenRows.join("/") + " " + sideToChar[board.sideToMove];
};

// src/board/board.ts
var ATTACKERS_MAX = 24;
var DEFENDERS_MAX = 12;
function createBoard() {
  return {
    board: new Int8Array(SQS).fill(0),
    attackers: new Int16Array(ATTACKERS_MAX).fill(HOLE),
    defenders: new Int16Array(DEFENDERS_MAX).fill(HOLE),
    kingSq: HOLE,
    attackersCount: 0,
    defendersCount: 0,
    rowOcc: new Uint16Array(BOARD_SIZE),
    colOcc: new Uint16Array(BOARD_SIZE),
    pieceIndexBySquare: new Int16Array(SQS).fill(HOLE),
    sideToMove: 0 /* ATTACKERS */,
    moveNumber: 1,
    zobrist: 0n,
    repTable: /* @__PURE__ */ new Map(),
    passThroughThrone: true,
    lastMoveTo: HOLE
  };
}
function setAttacker(board, sq) {
  if (board.attackersCount >= ATTACKERS_MAX) {
    throw new Error("Exceeded maximum attackers capacity");
  }
  board.attackers[board.attackersCount] = sq;
  board.attackersCount++;
  board.pieceIndexBySquare[sq] = board.attackersCount - 1;
}
function setDefender(board, sq) {
  if (board.defendersCount >= DEFENDERS_MAX) {
    throw new Error("Exceeded maximum defenders capacity");
  }
  board.defenders[board.defendersCount] = sq;
  board.defendersCount++;
  board.pieceIndexBySquare[sq] = board.defendersCount - 1;
}
function setPiece(board, sq, piece) {
  board.board[sq] = piece;
  board.zobrist ^= zobrist[piece - 1][sq];
  const row = getRow(sq);
  const col = getCol(sq);
  board.rowOcc[row] |= 1 << col;
  board.colOcc[col] |= 1 << row;
  if (piece === 1 /* ATTACKER */) {
    setAttacker(board, sq);
  } else if (piece === 2 /* DEFENDER */) {
    setDefender(board, sq);
  } else if (piece === 3 /* KING */) {
    board.kingSq = sq;
  }
}
function clearAttacker(board, sq) {
  const index = board.pieceIndexBySquare[sq];
  if (index !== -1) {
    if (board.attackersCount === 1) {
      board.attackers[index] = HOLE;
      board.pieceIndexBySquare[sq] = 0;
    } else {
      const lastSq = board.attackers[board.attackersCount - 1];
      board.attackers[index] = lastSq;
      board.attackers[board.attackersCount - 1] = HOLE;
      board.pieceIndexBySquare[lastSq] = index;
      board.pieceIndexBySquare[sq] = HOLE;
    }
    board.attackersCount--;
  }
}
function clearDefender(board, sq) {
  const index = board.pieceIndexBySquare[sq];
  if (index !== -1) {
    if (board.defendersCount === 1) {
      board.defenders[index] = HOLE;
      board.pieceIndexBySquare[sq] = 0;
    } else {
      const lastSq = board.defenders[board.defendersCount - 1];
      board.defenders[index] = lastSq;
      board.defenders[board.defendersCount - 1] = HOLE;
      board.pieceIndexBySquare[lastSq] = index;
      board.pieceIndexBySquare[sq] = HOLE;
    }
    board.defendersCount--;
  }
}
function clearPiece(board, sq) {
  const piece = board.board[sq];
  board.zobrist ^= zobrist[piece - 1][sq];
  board.board[sq] = 0 /* EMPTY */;
  const row = getRow(sq);
  const col = getCol(sq);
  board.rowOcc[row] &= ~(1 << col);
  board.colOcc[col] &= ~(1 << row);
  if (piece === 1 /* ATTACKER */) {
    clearAttacker(board, sq);
  } else if (piece === 2 /* DEFENDER */) {
    clearDefender(board, sq);
  } else if (piece === 3 /* KING */) {
    board.kingSq = HOLE;
  }
}
function flipSide(board) {
  board.sideToMove = board.sideToMove === 0 /* ATTACKERS */ ? 1 /* DEFENDERS */ : 0 /* ATTACKERS */;
  board.zobrist ^= sideZobrist;
}
function clearBoard(board) {
  board.board.fill(0 /* EMPTY */);
  board.attackers.fill(HOLE);
  board.defenders.fill(HOLE);
  board.kingSq = HOLE;
  board.attackersCount = 0;
  board.defendersCount = 0;
  board.rowOcc.fill(0);
  board.colOcc.fill(0);
  board.pieceIndexBySquare.fill(HOLE);
  board.sideToMove = 0 /* ATTACKERS */;
  board.moveNumber = 1;
  board.zobrist = 0n;
  board.repTable.clear();
  board.lastMoveTo = HOLE;
}
function setInitialPosition(board) {
  setFEN(board, INITIAL_FEN);
}

// src/moves/move/move.ts
function createMove(from, to) {
  return from << 16 | to;
}
function moveFrom(move) {
  return move >> 16 & 65535;
}
function moveTo(move) {
  return move & 65535;
}
function createMoveFromAlgebraic(move) {
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

// src/utils/utils.ts
function getBinary(num, digits = BOARD_SIZE, reverse = true) {
  const masked = num & (1 << digits) - 1;
  const res = masked.toString(2).padStart(digits, "0");
  return reverse ? res.split("").reverse().join("") : res;
}
function ctz(x) {
  return Math.clz32(x & -x) ^ 31;
}

// src/moves/attackers.ts
var LINE_MOVES = Array.from({ length: BOARD_SIZE }, () => new Uint16Array(1 << BOARD_SIZE));
function buildLineMask(at, occ) {
  let mask = 0;
  for (let i = at - 1; i >= 0; i--) {
    if (occ & 1 << i) break;
    mask |= 1 << i;
  }
  for (let i = at + 1; i < BOARD_SIZE; i++) {
    if (occ & 1 << i) break;
    mask |= 1 << i;
  }
  return mask;
}
function initLineMoves() {
  for (let pos = 0; pos < BOARD_SIZE; pos++) {
    for (let occ = 0; occ < 1 << BOARD_SIZE; occ++) {
      LINE_MOVES[pos][occ] = buildLineMask(pos, occ);
    }
  }
}

// src/moves/movegen/movegen.ts
var THRONE_MASK = 1 << 5;
var BOUNDARY_MASK = 1 | 1 << 10;
var disableBlockedSquares = (beam, piece, rowOrColIndex) => {
  if (piece === 1 /* ATTACKER */ || piece === 2 /* DEFENDER */) {
    if (rowOrColIndex === 5) {
      beam &= ~THRONE_MASK;
    } else if (rowOrColIndex === 0 || rowOrColIndex === 10) {
      beam &= ~BOUNDARY_MASK;
    }
  }
  return beam;
};
var createMoveGenerator = () => {
  const MAX_MOVES = 1024;
  const moves = new Uint32Array(MAX_MOVES);
  let moveCount = 0;
  const addMove = (fromSq, toSq) => {
    if (moveCount >= MAX_MOVES) {
      throw new Error("Exceeded maximum move capacity");
    }
    moves[moveCount++] = createMove(fromSq, toSq);
  };
  const generateHorizontalMoves = (board, fromSq) => {
    const row = getRow(fromSq);
    const col = getCol(fromSq);
    const rowOcc = board.rowOcc[row];
    let horizontalMoves = LINE_MOVES[col][rowOcc] & ~(1 << col);
    horizontalMoves = disableBlockedSquares(horizontalMoves, board.board[fromSq], row);
    let cur = 1;
    while (horizontalMoves) {
      if (cur & horizontalMoves) {
        const toSq = getSquare(row, ctz(cur));
        addMove(fromSq, toSq);
      }
      horizontalMoves &= ~cur;
      cur <<= 1;
    }
  };
  const generateVerticalMoves = (board, fromSq) => {
    const col = getCol(fromSq);
    const row = getRow(fromSq);
    const colOcc = board.colOcc[col];
    let verticalMoves = LINE_MOVES[row][colOcc] & ~(1 << row);
    verticalMoves = disableBlockedSquares(verticalMoves, board.board[fromSq], col);
    let cur = 1;
    while (verticalMoves) {
      if (cur & verticalMoves) {
        const toSq = getSquare(ctz(cur), col);
        addMove(fromSq, toSq);
      }
      verticalMoves &= ~cur;
      cur <<= 1;
    }
  };
  const generateMovesForPiece = (board, fromSq) => {
    if (fromSq === -1) return;
    generateHorizontalMoves(board, fromSq);
    generateVerticalMoves(board, fromSq);
  };
  const generateDefenderMoves = (board) => {
    for (let i = 0; i < board.defendersCount; i++) {
      if (board.defenders[i] === -1) continue;
      generateMovesForPiece(board, board.defenders[i]);
    }
    generateMovesForPiece(board, board.kingSq);
  };
  const generateAttackerMoves = (board) => {
    for (let i = 0; i < board.attackersCount; i++) {
      if (board.attackers[i] === -1) continue;
      generateMovesForPiece(board, board.attackers[i]);
    }
  };
  const movegen = (board) => {
    moveCount = 0;
    if (board.sideToMove === 0 /* ATTACKERS */) {
      generateAttackerMoves(board);
    } else {
      generateDefenderMoves(board);
    }
  };
  return {
    movegen,
    moves,
    get movesCount() {
      return moveCount;
    }
  };
};

// src/moves/makeMove/isCapturePossible/isCapturePossible.ts
var enemySquares = [...getCornersSq(), getThroneSq()];
var isCapturePossible = (board, targetSq, enemySq1, enemySq2) => {
  const attackerPiece1 = board.board[enemySq1];
  const attackerPiece2 = board.board[enemySq2];
  const attackerPiece = attackerPiece1 || attackerPiece2;
  const targetPieceForAttackers = board.board[targetSq];
  console.log({ attackerPiece1, attackerPiece2 });
  if (attackerPiece1 && attackerPiece2 && (attackerPiece1 !== attackerPiece2 && !(attackerPiece1 === 3 /* KING */ && attackerPiece2 === 2 /* DEFENDER */) && !(attackerPiece2 === 3 /* KING */ && attackerPiece1 === 2 /* DEFENDER */))) {
    return false;
  }
  if (targetPieceForAttackers === 0 /* EMPTY */ || targetPieceForAttackers === 3 /* KING */) {
    return false;
  }
  const attackerSide = getSideByPiece(attackerPiece);
  const targetPieceSide = getSideByPiece(targetPieceForAttackers);
  if (attackerPiece1 && attackerPiece2) {
    return attackerSide !== targetPieceSide;
  }
  if (!targetPieceForAttackers || targetPieceForAttackers === 3 /* KING */) {
    return false;
  }
  if (enemySquares.includes(enemySq1)) {
    return attackerSide !== targetPieceSide;
  }
  if (enemySquares.includes(enemySq2)) {
    return attackerSide !== targetPieceSide;
  }
  return false;
};

// src/moves/makeMove/makeShieldCaptures/makeShieldCaptures.ts
var shieldIterators = {
  [0 /* TOP */]: {
    getRoofNeighbor: getBottomNeighbor,
    next: getRightNeighbor,
    isLast: (sq) => sq === getTopRightSquare(),
    getStart: getTopLeftSquare,
    getNextNeighbor: getRightNeighbor,
    getPrevNeighbor: getLeftNeighbor,
    isAlwaysFriendSq: (sq) => sq === getTopLeftSquare() || sq === getTopRightSquare()
  },
  [1 /* BOTTOM */]: {
    getRoofNeighbor: getTopNeighbor,
    next: getRightNeighbor,
    isLast: (sq) => sq === getBottomRightSquare(),
    getStart: getBottomLeftSquare,
    getNextNeighbor: getRightNeighbor,
    getPrevNeighbor: getLeftNeighbor,
    isAlwaysFriendSq: (sq) => sq === getBottomLeftSquare() || sq === getBottomRightSquare()
  },
  [2 /* LEFT */]: {
    getRoofNeighbor: getRightNeighbor,
    next: getBottomNeighbor,
    isLast: (sq) => sq === getBottomLeftSquare(),
    getStart: getTopLeftSquare,
    getNextNeighbor: getBottomNeighbor,
    getPrevNeighbor: getTopNeighbor,
    isAlwaysFriendSq: (sq) => sq === getTopLeftSquare() || sq === getBottomLeftSquare()
  },
  [3 /* RIGHT */]: {
    getRoofNeighbor: getLeftNeighbor,
    next: getBottomNeighbor,
    isLast: (sq) => sq === getBottomRightSquare(),
    getStart: getTopRightSquare,
    getNextNeighbor: getBottomNeighbor,
    getPrevNeighbor: getTopNeighbor,
    isAlwaysFriendSq: (sq) => sq === getTopRightSquare() || sq === getBottomRightSquare()
  }
};
var getShieldSides = (toSq) => {
  const sides = [];
  const row = getRow(toSq);
  const col = getCol(toSq);
  if (row <= 1) {
    sides.push(1 /* BOTTOM */);
  }
  if (row >= BOARD_SIZE - 2) {
    sides.push(0 /* TOP */);
  }
  if (col <= 1) {
    sides.push(2 /* LEFT */);
  }
  if (col >= BOARD_SIZE - 2) {
    sides.push(3 /* RIGHT */);
  }
  return sides;
};
var getCapturesOnShieldSide = (board, toSq, side, iterator) => {
  let it = iterator.getStart(toSq);
  const res = [];
  let captureSequence = [];
  let seqStarted = true;
  const isFriend = (sq) => {
    return getSideByPiece(board.board[sq]) === side;
  };
  const resetSequence = () => {
    captureSequence = [];
    seqStarted = false;
  };
  while (it = iterator.next(it)) {
    if (iterator.isLast(it)) {
      if (captureSequence.length > 1) {
        res.push(...captureSequence);
        captureSequence = [];
      }
    } else if (board.board[it] === 0 /* EMPTY */) {
      resetSequence();
    } else if (isFriend(it)) {
      if (captureSequence.length > 1) {
        res.push(...captureSequence);
        captureSequence = [];
      }
      seqStarted = true;
    } else {
      if (!seqStarted) {
        continue;
      }
      const roof = iterator.getRoofNeighbor(it);
      if (getSideByPiece(board.board[roof]) === side) {
        captureSequence.push(it);
      } else {
        resetSequence();
      }
    }
  }
  return res;
};
var makeShieldWallCaptures = (board, toSq, side) => {
  const res = [];
  const sides = getShieldSides(toSq);
  if (sides.length === 0) {
    return [];
  }
  sides.forEach((shieldSide) => {
    const captures = getCapturesOnShieldSide(
      board,
      toSq,
      side,
      shieldIterators[shieldSide]
    );
    for (const sq of captures) {
      if (board.board[sq] !== 3 /* KING */) {
        res.push({
          sq,
          piece: board.board[sq]
        });
        clearPiece(board, sq);
      }
    }
  });
  return res;
};

// src/moves/makeMove/makeMove.ts
var getSquareByPath = (sq, right, top) => {
  const row = getRow(sq);
  const col = getCol(sq);
  const newRow = row + top;
  const newCol = col + right;
  if (newRow < 0 || newRow >= BOARD_SIZE || newCol < 0 || newCol >= BOARD_SIZE) {
    return -1;
  }
  return getSquare(newRow, newCol);
};
var getPotentialPairs = (sq) => {
  return [
    getSquareByPath(sq, 2, 0),
    // right
    getSquareByPath(sq, -2, 0),
    // left
    getSquareByPath(sq, 0, 2),
    // up
    getSquareByPath(sq, 0, -2)
    // down
  ].filter((s) => s !== -1);
};
var getBetweenSquare = (sq1, sq2) => {
  const row1 = getRow(sq1);
  const col1 = getCol(sq1);
  const row2 = getRow(sq2);
  const col2 = getCol(sq2);
  if (row1 === row2 && Math.abs(col1 - col2) === 2) {
    return getSquare(row1, (col1 + col2) / 2);
  }
  if (col1 === col2 && Math.abs(row1 - row2) === 2) {
    return getSquare((row1 + row2) / 2, col1);
  }
  return -1;
};
var addPositionToRepTable = (board) => {
  if (board.repTable.has(board.zobrist)) {
    board.repTable.set(board.zobrist, board.repTable.get(board.zobrist) + 1);
  } else {
    board.repTable.set(board.zobrist, 1);
  }
};
var makeMove = (board, move) => {
  const fromSq = moveFrom(move);
  const toSq = moveTo(move);
  const piece = board.board[fromSq];
  const lastMoveTo = board.lastMoveTo;
  clearPiece(board, fromSq);
  setPiece(board, toSq, piece);
  const potentialPairs = getPotentialPairs(toSq);
  const captured = [];
  for (const pairSq of potentialPairs) {
    const betweenSq = getBetweenSquare(toSq, pairSq);
    if (betweenSq === -1) {
      continue;
    }
    if (isCapturePossible(board, betweenSq, toSq, pairSq)) {
      captured.push({
        sq: betweenSq,
        piece: board.board[betweenSq]
      });
      clearPiece(board, betweenSq);
    }
  }
  const shieldWallCaptures = makeShieldWallCaptures(board, toSq, board.sideToMove);
  flipSide(board);
  board.lastMoveTo = toSq;
  addPositionToRepTable(board);
  return {
    from: fromSq,
    to: toSq,
    captured: shieldWallCaptures.length > 0 ? [...captured, ...shieldWallCaptures] : captured,
    movedPiece: piece,
    lastMoveTo
  };
};

// src/moves/unmakeMove/unmakeMove.ts
function removePositionFromRepTable(board) {
  if (board.repTable.has(board.zobrist)) {
    board.repTable.set(board.zobrist, board.repTable.get(board.zobrist) - 1);
  }
}
function unmakeMove(board, undo) {
  removePositionFromRepTable(board);
  clearPiece(board, undo.to);
  setPiece(board, undo.from, undo.movedPiece);
  for (const captured of undo.captured) {
    setPiece(board, captured.sq, captured.piece);
  }
  flipSide(board);
  board.lastMoveTo = undo.lastMoveTo;
}

// src/moves/init.ts
function initMovesModule() {
  initLineMoves();
}

// src/moves/utils.ts
function getMoveAlg(move) {
  return `${getSquareAlg(moveFrom(move))}${getSquareAlg(moveTo(move))}`;
}

// src/evaluation/constants.ts
var MATE_SCORE = 1e4;

// src/evaluation/utils.ts
var MATE_SCORE_THRESHOLD = MATE_SCORE - 256;
var isMateScore = (score) => {
  return Math.abs(score) >= MATE_SCORE_THRESHOLD;
};

// src/evaluation/getScoreText.ts
var getScoreText = (score) => {
  if (isMateScore(score)) {
    return `mate in ${MATE_SCORE - Math.abs(score)}`;
  }
  return String(score);
};

// src/evaluation/kingIsSurrounded/kingIsSurrounded.ts
var THRONE_SQ = getThroneSq();
var kingIsSurrounded = (board) => {
  const { kingSq } = board;
  if (board.kingSq === -1) {
    throw new Error("King square is not defined");
  }
  return [
    getLeftNeighbor(kingSq),
    getTopNeighbor(kingSq),
    getRightNeighbor(kingSq),
    getBottomNeighbor(kingSq)
  ].filter((sq) => {
    if (sq === 0 /* EMPTY */) {
      return false;
    }
    if (sq === THRONE_SQ) {
      return true;
    }
    return board.board[sq] === 1 /* ATTACKER */;
  }).length === 4;
};

// src/utils/queue/createQueue.ts
var createQueue = (startSquares) => {
  const queue = [...startSquares];
  let head = 0;
  return {
    enqueue: (value) => {
      queue.push(value);
    },
    dequeue: () => {
      return head < queue.length ? queue[head++] : void 0;
    },
    isEmpty: () => {
      return head >= queue.length;
    }
  };
};

// src/utils/bfs/bfs.ts
var bfs = ({ isAchievable, startSquares, getNeighbors = getVerticalHorizontalNeighbors }) => {
  const queue = createQueue(startSquares);
  const visitedFlags = new Uint8Array(SQS);
  while (!queue.isEmpty()) {
    const currentSq = queue.dequeue();
    const neighbors = getNeighbors(currentSq);
    for (let i = 0; i < neighbors.length; i++) {
      const neighbor = neighbors[i];
      if (visitedFlags[neighbor]) continue;
      if (!isAchievable(neighbor)) continue;
      visitedFlags[neighbor] = 1;
      queue.enqueue(neighbor);
    }
  }
  return visitedFlags;
};

// src/evaluation/defendersIsSurrounded/defendersIsSurrounded.ts
var corners = getCornersSq();
var isCalculateNeeded = (board) => {
  const lastMoveSq = board.lastMoveTo;
  if (lastMoveSq === HOLE) {
    return true;
  }
  if (board.board[lastMoveSq] !== 1 /* ATTACKER */) {
    return false;
  }
  const allSiblings = getAllNeighbors(lastMoveSq);
  let attackerNearbyCount = 0;
  for (let i = 0; i < allSiblings.length; i++) {
    const sq = allSiblings[i];
    if (board.board[sq] === 1 /* ATTACKER */) {
      attackerNearbyCount++;
      if (attackerNearbyCount >= 2) {
        return true;
      }
    }
  }
  return false;
};
var defendersIsSurrounded = (board) => {
  if (!isCalculateNeeded(board)) {
    return false;
  }
  const visited = bfs({
    isAchievable: (sq) => board.board[sq] !== 1 /* ATTACKER */,
    startSquares: corners
  });
  for (let i = 0; i < SQS; i++) {
    if (!visited[i]) {
      continue;
    }
    if (board.board[i] === 2 /* DEFENDER */ || board.board[i] === 3 /* KING */) {
      return false;
    }
  }
  const innerSquares = bfs({
    isAchievable: (sq) => board.board[sq] !== 1 /* ATTACKER */,
    startSquares: [board.kingSq]
  });
  for (let i = 0; i < SQS; i++) {
    if (innerSquares[i] && isEdgeSquare(i)) {
      return false;
    }
  }
  return true;
};

// src/utils/getSetFromBinary.ts
var getSetFromBinary = (array) => {
  const resultSet = /* @__PURE__ */ new Set();
  for (let i = 0; i < array.length; i++) {
    if (array[i]) {
      resultSet.add(i);
    }
  }
  return resultSet;
};

// src/evaluation/checkFort/checkFort.ts
var isCornerSquare = (sq) => {
  return sq === getTopLeftSquare() || sq === getBottomLeftSquare() || sq === getTopRightSquare() || sq === getBottomRightSquare();
};
var kingHasMoves = (board) => {
  const potentialMoves = getVerticalHorizontalNeighbors(board.kingSq);
  let possibleMoves = 0;
  for (let i = 0; i < potentialMoves.length; i++) {
    if (board.board[potentialMoves[i]] === 0 /* EMPTY */) {
      possibleMoves++;
    }
  }
  return possibleMoves > 0;
};
var kingContactEdges = (board) => {
  const { kingSq } = board;
  const row = getRow(kingSq);
  const col = getCol(kingSq);
  return row === 0 || row === BOARD_SIZE - 1 || col === 0 || col === BOARD_SIZE - 1;
};
var getFort = (board) => {
  const { kingSq } = board;
  let isAttackersAccessingKing = false;
  let isCornerAccessingKing = false;
  const fort = /* @__PURE__ */ new Set();
  const innerSpace = bfs({
    isAchievable: (sq) => {
      const piece = board.board[sq];
      if (piece === 1 /* ATTACKER */) {
        isAttackersAccessingKing = true;
      }
      if (isCornerSquare(sq)) {
        isCornerAccessingKing = true;
      }
      if (piece === 2 /* DEFENDER */) {
        fort.add(sq);
      }
      return piece === 0 /* EMPTY */;
    },
    startSquares: [kingSq]
  });
  innerSpace[kingSq] = 1;
  return {
    fort,
    isAttackersAccessingKing,
    isCornerAccessingKing,
    innerSpace
  };
};
var isFromOutside = (sq1, sq2, innerSpace) => {
  const isSq1Outside = innerSpace[sq1] === 0;
  const isSq2Outside = innerSpace[sq2] === 0;
  return isSq1Outside && isSq2Outside;
};
var couldHavePotentialAttacker = (board, sq) => {
  return board.board[sq] !== 2 /* DEFENDER */ && sq !== getThroneSq();
};
var isCapturePossibleInEnemyArea = (board, betweenA, betweenB, innerSpace) => {
  return betweenA !== null && betweenB !== null && isFromOutside(betweenA, betweenB, innerSpace) && couldHavePotentialAttacker(board, betweenA) && couldHavePotentialAttacker(board, betweenB);
};
var Stats = {
  calls: 0
};
var isFortBreakable = (board, fort, innerSpace, height = 0) => {
  Stats.calls++;
  if (fort.size === 0) {
    throw new Error("Fort is empty");
  }
  const fortSq = fort.values().next();
  if (fortSq.value === void 0) {
    throw new Error("Fort is empty");
  }
  let fullSurroundedSquares = bfs({
    isAchievable: (sq) => board.board[sq] === 2 /* DEFENDER */,
    startSquares: [fortSq.value],
    getNeighbors: getAllNeighbors
  });
  const fullSurroundedSquaresSet = getSetFromBinary(fullSurroundedSquares);
  const iterator = fullSurroundedSquaresSet.values();
  while (true) {
    const nextSquareToCheck = iterator.next();
    if (nextSquareToCheck.value === void 0) {
      break;
    }
    const [left, right] = [getLeftNeighbor(nextSquareToCheck.value), getRightNeighbor(nextSquareToCheck.value)];
    const [top, bottom] = [getTopNeighbor(nextSquareToCheck.value), getBottomNeighbor(nextSquareToCheck.value)];
    if (isCapturePossibleInEnemyArea(
      board,
      left,
      right,
      innerSpace
    )) {
      if (fort.has(nextSquareToCheck.value)) {
        return true;
      }
      clearPiece(board, nextSquareToCheck.value);
      const res = isFortBreakable(board, fort, innerSpace, height + 1);
      setPiece(board, nextSquareToCheck.value, 2 /* DEFENDER */);
      if (res) {
        return true;
      }
    }
    if (isCapturePossibleInEnemyArea(
      board,
      top,
      bottom,
      innerSpace
    )) {
      if (fort.has(nextSquareToCheck.value)) {
        return true;
      }
      clearPiece(board, nextSquareToCheck.value);
      const res = isFortBreakable(board, fort, innerSpace, height + 1);
      setPiece(board, nextSquareToCheck.value, 2 /* DEFENDER */);
      if (res) {
        return true;
      }
    }
  }
  return false;
};
var isCalculateNeeded2 = (board) => {
  const { lastMoveTo } = board;
  if (lastMoveTo === HOLE || board.board[lastMoveTo] === 3 /* KING */) {
    return true;
  }
  if (board.board[lastMoveTo] === 1 /* ATTACKER */) {
    return false;
  }
  const isEdge = isEdgeSquare(lastMoveTo);
  const allNeighbors = getAllNeighbors(lastMoveTo);
  let defendersNearbyCount = 0;
  for (let i = 0; i < allNeighbors.length; i++) {
    const sq = allNeighbors[i];
    if (board.board[sq] === 2 /* DEFENDER */ || isEdgeSquare(sq)) {
      defendersNearbyCount++;
    }
  }
  return isEdge && defendersNearbyCount > 0 || defendersNearbyCount >= 2;
};
var checkFort = (board) => {
  if (!isCalculateNeeded2(board)) {
    return false;
  }
  if (!kingHasMoves(board) || !kingContactEdges(board)) {
    return false;
  }
  const { fort, isAttackersAccessingKing, isCornerAccessingKing, innerSpace } = getFort(board);
  if (isAttackersAccessingKing || isCornerAccessingKing) {
    return false;
  }
  if (fort.size === 0) {
    return false;
  }
  return !isFortBreakable(board, fort, innerSpace);
};

// src/evaluation/terminal.ts
var cornersSquares = [
  getTopRightSquare(),
  getTopLeftSquare(),
  getBottomLeftSquare(),
  getBottomRightSquare()
];
var checkTerminal = (board) => {
  if (board.repTable.get(board.zobrist) >= 3) {
    return 0 /* ATTACKERS */;
  }
  if (cornersSquares.includes(board.kingSq) || board.attackersCount < 2) {
    return 1 /* DEFENDERS */;
  }
  if (board.attackersCount < 2) {
    return 1 /* DEFENDERS */;
  }
  if (kingIsSurrounded(board)) {
    return 0 /* ATTACKERS */;
  }
  if (defendersIsSurrounded(board)) {
    return 0 /* ATTACKERS */;
  }
  if (checkFort(board)) {
    return 1 /* DEFENDERS */;
  }
  return null;
};

// src/evaluation/evaluate.ts
var sidedEval = (board, score) => {
  return board.sideToMove === 1 /* DEFENDERS */ ? score : -score;
};
var evaluateBoard = (board) => {
  const { attackersCount, defendersCount } = board;
  return sidedEval(board, defendersCount - attackersCount / 2);
};

// src/search/model/BestMove.ts
var BestMove = class {
  move = 0;
  setBestMove(move) {
    this.move = move;
  }
};
var bestMove = new BestMove();

// src/search/search.ts
var MAX_DEPTH = 256;
var moveGens = Array.from({ length: MAX_DEPTH }, () => createMoveGenerator());
var moveGenAtDepth = (depth) => {
  return moveGens[depth];
};
var search = (board, depth, alpha = -MATE_SCORE, beta = MATE_SCORE, height = 0) => {
  if (height === 0) {
    statistics.reset();
  }
  const terminal = checkTerminal(board);
  if (terminal !== null) {
    return sidedEval(board, terminal === 1 /* DEFENDERS */ ? MATE_SCORE - height : -MATE_SCORE + height);
  }
  if (depth === 0) {
    return evaluateBoard(board);
  }
  const moveGen = moveGenAtDepth(depth);
  moveGen.movegen(board);
  for (let i = 0; i < moveGen.movesCount; i++) {
    const move = moveGen.moves[i];
    statistics.incrementNodes();
    const undo = makeMove(board, move);
    const score = -search(
      board,
      depth - 1,
      -beta,
      -alpha,
      height + 1
    );
    unmakeMove(board, undo);
    if (score > alpha) {
      alpha = score;
      if (height === 0) {
        bestMove.setBestMove(move);
      }
    }
    if (alpha >= beta) {
      break;
    }
  }
  return alpha;
};
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  BOARD_SIZE,
  HOLE,
  INITIAL_FEN,
  MATE_SCORE,
  NUM_PIECE_KINDS,
  Piece,
  SQS,
  Side,
  bestMove,
  bfs,
  checkTerminal,
  clearBoard,
  clearPiece,
  createBoard,
  createMove,
  createMoveFromAlgebraic,
  createMoveGenerator,
  ctz,
  evaluateBoard,
  getBinary,
  getBottomLeftSquare,
  getBottomRightSquare,
  getCol,
  getCornersSq,
  getFEN,
  getMoveAlg,
  getOppositeSide,
  getRow,
  getScoreText,
  getSideByPiece,
  getSquare,
  getSquareAlg,
  getSquareFromAlgebraic,
  getThroneSq,
  getTopLeftSquare,
  getTopRightSquare,
  initMovesModule,
  isMateScore,
  makeMove,
  moveFrom,
  moveTo,
  search,
  setFEN,
  setInitialPosition,
  setPiece,
  sidedEval,
  statistics,
  unmakeMove
});
//# sourceMappingURL=index.cjs.map