declare class Statistics {
    nodes: number;
    reset(): void;
    incrementNodes(count?: number): void;
}
declare const statistics: Statistics;

declare enum Piece {
    EMPTY = 0,
    ATTACKER = 1,
    DEFENDER = 2,
    KING = 3
}
declare enum Side {
    ATTACKERS = 0,
    DEFENDERS = 1
}

interface Board {
    board: Int8Array;
    attackers: Int16Array;
    defenders: Int16Array;
    kingSq: number;
    attackersCount: number;
    defendersCount: number;
    pieceIndexBySquare: Int16Array;
    rowOcc: Uint16Array;
    colOcc: Uint16Array;
    sideToMove: Side.ATTACKERS | Side.DEFENDERS;
    moveNumber: number;
    zobrist: bigint;
    repTable: Map<bigint, number>;
    passThroughThrone: boolean;
    lastMoveTo: number;
}

declare function createBoard(): Board;
declare function setPiece(board: Board, sq: number, piece: number): void;
declare function clearPiece(board: Board, sq: number): void;
declare function clearBoard(board: Board): void;
declare function setInitialPosition(board: Board): void;

declare function getSquare(row: number, col: number): number;
declare function getSquareFromAlgebraic(coord: string): number;
declare function getRow(sq: number): number;
declare function getCol(sq: number): number;
declare function getSideByPiece(piece: number): Side | null;
declare const getThroneSq: () => number;
declare const getCornersSq: () => number[];
declare function getSquareAlg(sq: number): string;
declare function getOppositeSide(side: Side): Side;
declare function getTopLeftSquare(): number;
declare function getBottomRightSquare(): number;
declare function getTopRightSquare(): number;
declare function getBottomLeftSquare(): number;

declare const BOARD_SIZE = 11;
declare const SQS: number;
declare const HOLE = -1;
declare const NUM_PIECE_KINDS = 3;
declare const INITIAL_FEN = "3aaaaa3/5a5/11/a4d4a/a3ddd3a/aa1ddkdd1aa/a3ddd3a/a4d4a/11/5a5/3aaaaa3 a";

declare const setFEN: (board: Board, fen: string) => void;
declare const getFEN: (board: Board) => string;

declare const search: (board: Board, depth: number, alpha?: number, beta?: number, height?: number) => number;

declare class BestMove {
    move: number;
    setBestMove(move: number): void;
}
declare const bestMove: BestMove;

interface Params {
    onIteration?: (depth: number, move: number, score: number, nodes: number, speed: number, time: number) => void;
    time: number;
}
declare const searchRoot: (board: Board, { onIteration, time }: Params) => {
    bestMove: number;
    bestScore: number;
};

declare function getBinary(num: number, digits?: number, reverse?: boolean): string;
declare function ctz(x: number): number;
declare function setMoves(board: Board, moves: number[]): void;

interface Options {
    isAchievable: (sq: number) => boolean;
    startSquares: number[];
    getNeighbors?: (sq: number) => number[];
}
declare const bfs: ({ isAchievable, startSquares, getNeighbors }: Options) => Uint8Array<ArrayBuffer>;

declare const MATE_SCORE = 10000;

declare const getScoreText: (score: number) => string;

declare const checkTerminal: (board: Board) => Side | null;

declare const isMateScore: (score: number) => boolean;

declare const sidedEval: (board: Board, score: number) => number;
declare const evaluateBoard: (board: Board) => number;

declare function createMove(from: number, to: number): number;
declare function moveFrom(move: number): number;
declare function moveTo(move: number): number;
declare function createMoveFromAlgebraic(move: string): number;

interface MoveGenerator {
    movegen: (board: Board) => void;
    moves: Uint32Array;
    movesCount: number;
    MAX_MOVES: number;
    decreaseCount: () => void;
}
declare const createMoveGenerator: () => MoveGenerator;

interface CapturedPiece {
    sq: number;
    piece: Piece;
}
interface UndoMove {
    from: number;
    to: number;
    captured: CapturedPiece[];
    movedPiece: Piece;
    lastMoveTo: number;
}

declare const makeMove: (board: Board, move: number) => UndoMove;

declare function unmakeMove(board: Board, undo: UndoMove): void;

declare function initMovesModule(): void;

declare function getMoveAlg(move: number): string;

export { BOARD_SIZE, type Board, HOLE, INITIAL_FEN, MATE_SCORE, NUM_PIECE_KINDS, Piece, SQS, Side, bestMove, bfs, checkTerminal, clearBoard, clearPiece, createBoard, createMove, createMoveFromAlgebraic, createMoveGenerator, ctz, evaluateBoard, getBinary, getBottomLeftSquare, getBottomRightSquare, getCol, getCornersSq, getFEN, getMoveAlg, getOppositeSide, getRow, getScoreText, getSideByPiece, getSquare, getSquareAlg, getSquareFromAlgebraic, getThroneSq, getTopLeftSquare, getTopRightSquare, initMovesModule, isMateScore, makeMove, moveFrom, moveTo, search, searchRoot, setFEN, setInitialPosition, setMoves, setPiece, sidedEval, statistics, unmakeMove };
