import {
  Board,
  BOARD_SIZE, BOTTOM_NEIGHBOR, clearPiece,
  getBottomLeftSquare,
  getBottomRightSquare,
  getCol,
  getRow,
  getSideByPiece,
  getTopLeftSquare,
  getTopRightSquare, LEFT_NEIGHBOR,
  Piece, RIGHT_NEIGHBOR,
  Side, TOP_NEIGHBOR
} from "@/board";
import {CapturedPiece} from "@/moves/model/UndoMove";

const getBottomNeighbor = (sq: number) => BOTTOM_NEIGHBOR[sq];
const getTopNeighbor = (sq: number) => TOP_NEIGHBOR[sq];
const getLeftNeighbor = (sq: number) => LEFT_NEIGHBOR[sq];
const getRightNeighbor = (sq: number) => RIGHT_NEIGHBOR[sq];

interface ShieldIterator {
  getRoofNeighbor: (sq: number) => number | null;
  next: (sq: number) => number | null;
  isLast: (sq: number | null) => boolean;
  getStart: (sq: number | null) => number;
  getNextNeighbor: (sq: number) => number | null;
  getPrevNeighbor: (sq: number) => number | null;
  isAlwaysFriendSq: (sq: number) => boolean;
}

const enum ShieldSides {
  TOP,
  BOTTOM,
  LEFT,
  RIGHT,
}

const shieldIterators: Record<ShieldSides, ShieldIterator> = {
  [ShieldSides.TOP]: {
    getRoofNeighbor: getBottomNeighbor,
    next: getRightNeighbor,
    isLast: (sq) => sq === getTopRightSquare(),
    getStart: getTopLeftSquare,
    getNextNeighbor: getRightNeighbor,
    getPrevNeighbor: getLeftNeighbor,
    isAlwaysFriendSq: (sq: number) => sq === getTopLeftSquare() || sq === getTopRightSquare(),
  },
  [ShieldSides.BOTTOM]: {
    getRoofNeighbor: getTopNeighbor,
    next: getRightNeighbor,
    isLast: (sq) => sq === getBottomRightSquare(),
    getStart: getBottomLeftSquare,
    getNextNeighbor: getRightNeighbor,
    getPrevNeighbor: getLeftNeighbor,
    isAlwaysFriendSq: (sq: number) => sq === getBottomLeftSquare() || sq === getBottomRightSquare(),
  },
  [ShieldSides.LEFT]: {
    getRoofNeighbor: getRightNeighbor,
    next: getBottomNeighbor,
    isLast: (sq) => sq === getBottomLeftSquare(),
    getStart: getTopLeftSquare,
    getNextNeighbor: getBottomNeighbor,
    getPrevNeighbor: getTopNeighbor,
    isAlwaysFriendSq: (sq: number) => sq === getTopLeftSquare() || sq === getBottomLeftSquare(),
  },
  [ShieldSides.RIGHT]: {
    getRoofNeighbor: getLeftNeighbor,
    next: getBottomNeighbor,
    isLast: (sq) => sq === getBottomRightSquare(),
    getStart: getTopRightSquare,
    getNextNeighbor: getBottomNeighbor,
    getPrevNeighbor: getTopNeighbor,
    isAlwaysFriendSq: (sq: number) => sq === getTopRightSquare() || sq === getBottomRightSquare(),
  },
};

const getShieldSides = (toSq: number) => {
  const sides: ShieldSides[] = []

  const row = getRow(toSq)
  const col = getCol(toSq)

  if (row <= 1) {
    sides.push(ShieldSides.BOTTOM)
  }

  if (row >= BOARD_SIZE - 2) {
    sides.push(ShieldSides.TOP)
  }

  if (col <= 1) {
    sides.push(ShieldSides.LEFT)
  }

  if (col >= BOARD_SIZE - 2) {
    sides.push(ShieldSides.RIGHT)
  }

  return sides;
}

const getCapturesOnShieldSide = (
  board: Board,
  toSq: number,
  side: Side,
  iterator: ShieldIterator
): number[] => {
  let it: number | null = iterator.getStart(toSq)

  const res = []
  let captureSequence: number[] = []
  let seqStarted = true

  const isFriend = (sq: number) => {
    return getSideByPiece(board.board[sq]) === side
  }

  const resetSequence = () => {
    captureSequence = []
    seqStarted = false
  }

  while (it = iterator.next(it!)) {
    if (iterator.isLast(it)) {
      if (captureSequence.length > 1) {
        res.push(...captureSequence)
        captureSequence = []
      }
    } else if (board.board[it] === Piece.EMPTY) {
      resetSequence()
    } else if (isFriend(it)) {
      if (captureSequence.length > 1) {
        res.push(...captureSequence)
        captureSequence = []
      }
      seqStarted = true
    } else {
      if (!seqStarted) {
        continue
      }

      const roof = iterator.getRoofNeighbor(it)

      if (getSideByPiece(board.board[roof!]) === side) {
        captureSequence.push(it)
      } else {
        resetSequence()
      }
    }
  }

  return res
}

export const makeShieldWallCaptures = (
  board: Board,
  toSq: number,
  side: Side
): CapturedPiece[] => {
  const res: CapturedPiece[] = []
  const sides = getShieldSides(toSq)

  if (sides.length === 0) {
    return []
  }

  sides.forEach((shieldSide) => {
    const captures = getCapturesOnShieldSide(
      board,
      toSq,
      side,
      shieldIterators[shieldSide]
    )

    for (const sq of captures) {
      if (board.board[sq] !== Piece.KING) {
        res.push({
          sq,
          piece: board.board[sq],
        })
        clearPiece(board, sq);
      }
    }
  })

  return res;
}
