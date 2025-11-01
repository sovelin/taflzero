import {
  Board,
  BOARD_SIZE, clearPiece,
  getBottomLeftSquare,
  getBottomRightSquare,
  getCol,
  getRow, getThroneSq,
  getTopLeftSquare,
  getTopRightSquare, HOLE,
  Piece, setPiece
} from "@/board";
import {bfs} from "@/utils";
import {
  getAllNeighbors, getBottomNeighbor,
  getLeftNeighbor,
  getRightNeighbor,
  getTopNeighbor,
  getVerticalHorizontalNeighbors, isEdgeSquare
} from "@/board/utils";
import {getSetFromBinary} from "@/utils/getSetFromBinary";

const isCornerSquare = (sq: number) => {
  return sq === getTopLeftSquare()
    || sq === getBottomLeftSquare()
    || sq === getTopRightSquare()
    || sq === getBottomRightSquare();
}

const kingHasMoves = (board: Board)=> {
  const potentialMoves = getVerticalHorizontalNeighbors(board.kingSq)

  let possibleMoves = 0
  for (let i = 0; i < potentialMoves.length; i++) {
    if (board.board[potentialMoves[i]] === Piece.EMPTY) {
      possibleMoves++;
    }
  }

  return possibleMoves > 0;
}

const kingContactEdges = (board: Board) => {
  const {kingSq} = board;
  const row = getRow(kingSq)
  const col = getCol(kingSq)

  return row === 0 || row === BOARD_SIZE - 1 || col === 0 || col === BOARD_SIZE - 1;
}

const getFort = (board: Board) => {
  const {kingSq} = board;
  let isAttackersAccessingKing = false
  let isCornerAccessingKing = false
  const fort = new Set<number>();

  const innerSpace = bfs({
    isAchievable: (sq: number) => {
      const piece = board.board[sq];

      if (piece === Piece.ATTACKER) {
        isAttackersAccessingKing = true;
      }

      if (isCornerSquare(sq)) {
        isCornerAccessingKing = true;
      }

      if (piece === Piece.DEFENDER) {
        fort.add(sq);
      }

      return piece === Piece.EMPTY;
    },
    startSquares: [kingSq]
  })

  innerSpace[kingSq] = 1;

  return {
    fort,
    isAttackersAccessingKing,
    isCornerAccessingKing,
    innerSpace,
  }
}

const isFromOutside = (sq1: number, sq2: number, innerSpace: Uint8Array) => {
  const isSq1Outside = innerSpace[sq1] === 0;
  const isSq2Outside = innerSpace[sq2] === 0;

  return isSq1Outside && isSq2Outside;
}

const couldHavePotentialAttacker = (board: Board, sq: number) => {
  return board.board[sq] !== Piece.DEFENDER && sq !== getThroneSq()
}

const isCapturePossibleInEnemyArea = (
  board: Board,
  betweenA: number | null,
  betweenB: number | null,
  innerSpace: Uint8Array,
) => {
  return betweenA !== null
  && betweenB !== null
  && isFromOutside(betweenA, betweenB, innerSpace)
  && couldHavePotentialAttacker(board, betweenA)
  && couldHavePotentialAttacker(board, betweenB)
}

export const Stats = {
  calls: 0,
}

const isFortBreakable = (board: Board, fort: Set<number>, innerSpace: Uint8Array, height = 0) => {
  Stats.calls++
  if (fort.size === 0) {
    throw new Error('Fort is empty');
  }

  const fortSq = fort.values().next()

  if (fortSq.value === undefined) {
    throw new Error('Fort is empty');
  }


  let fullSurroundedSquares = bfs({
    isAchievable: (sq: number) => board.board[sq] === Piece.DEFENDER,
    startSquares: [fortSq.value],
    getNeighbors: getAllNeighbors,
  })


  const fullSurroundedSquaresSet = getSetFromBinary(fullSurroundedSquares)

  /* Go through all fullSurroundedSquares and check can we break it or not
  * If it not possible, we stop process
  */
  const iterator = fullSurroundedSquaresSet.values();
  while (true) {
    const nextSquareToCheck = iterator.next();

    if (nextSquareToCheck.value === undefined) {
      break
    }

    const [left, right] = [getLeftNeighbor(nextSquareToCheck.value), getRightNeighbor(nextSquareToCheck.value)];
    const [top, bottom] = [getTopNeighbor(nextSquareToCheck.value), getBottomNeighbor(nextSquareToCheck.value)];

    if (isCapturePossibleInEnemyArea(
      board, left, right, innerSpace
    )) {
      if (fort.has(nextSquareToCheck.value)) {
        return true
      }

      clearPiece(board, nextSquareToCheck.value);
      const res = isFortBreakable(board, fort, innerSpace, height + 1);
      setPiece(board, nextSquareToCheck.value, Piece.DEFENDER);

      if (res) {
        return true;
      }
    }

    if (isCapturePossibleInEnemyArea(
      board, top, bottom, innerSpace
    )) {
      if (fort.has(nextSquareToCheck.value)) {
        return true
      }

      clearPiece(board, nextSquareToCheck.value);
      const res = isFortBreakable(board, fort, innerSpace, height + 1);
      setPiece(board, nextSquareToCheck.value, Piece.DEFENDER);

      if (res) {
        return true;
      }
    }
  }

  return false
}

const isCalculateNeeded = (board: Board) => {
  const {lastMoveTo} = board;

  if (lastMoveTo === HOLE || board.board[lastMoveTo] === Piece.KING) {
    return true;
  }

  if (board.board[lastMoveTo] === Piece.ATTACKER) {
    return false;
  }

  const isEdge = isEdgeSquare(lastMoveTo);
  const allNeighbors = getAllNeighbors(lastMoveTo);

  let defendersNearbyCount = 0;

  for (let i = 0; i < allNeighbors.length; i++) {
    const sq = allNeighbors[i];
    if (board.board[sq] === Piece.DEFENDER || isEdgeSquare(sq)) {
      defendersNearbyCount++;
    }
  }

  return isEdge && defendersNearbyCount > 0 || defendersNearbyCount >= 2;
}

export const checkFort = (board: Board) => {
  if (!isCalculateNeeded(board)) {
    return false;
  }

  if (!kingHasMoves(board) || !kingContactEdges(board)) {
    return false;
  }

  const {fort, isAttackersAccessingKing, isCornerAccessingKing, innerSpace} = getFort(board);

  if (isAttackersAccessingKing || isCornerAccessingKing) {
    return false;
  }

  if (fort.size === 0) {
    return false;
  }

  return !isFortBreakable(board, fort, innerSpace)
}
