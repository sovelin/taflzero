import {Board, getBottomLeftSquare, getBottomRightSquare, getTopLeftSquare, getTopRightSquare, Side} from "@/board";

const cornersSquares = [
  getTopRightSquare(),
  getTopLeftSquare(),
  getBottomLeftSquare(),
  getBottomRightSquare(),
];

export const checkTerminal = (board: Board): Side | null => {
  if (cornersSquares.includes(board.kingSq) || board.attackersCount < 2) {
    return Side.DEFENDERS
  }

  if (board.attackersCount < 2) {
    return Side.DEFENDERS
  }

  return null
}
