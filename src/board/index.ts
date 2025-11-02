export {createBoard, clearPiece, setPiece, clearBoard, setInitialPosition} from './board'
export type {Board} from './model/Board'
export {Piece, Side} from './types'
export {
  getCol,
  getCornersSq,
  getRow,
  getSideByPiece,
  getSquare,
  getThroneSq,
  getOppositeSide,
  getSquareAlg,
  getSquareFromAlgebraic,
  getTopLeftSquare,
  getBottomRightSquare,
  getTopRightSquare,
  getBottomLeftSquare,
} from './utils/utils'
export {BOARD_SIZE, HOLE, SQS, NUM_PIECE_KINDS, INITIAL_FEN} from './constants'
export {getFEN, setFEN} from './fen/fen'
