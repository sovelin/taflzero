import {Board, Piece} from "@/board";
import {bfs} from "@/utils";
import {getAllNeighbors} from "@/board/utils";

export const checkCycle = (board: Board, fromSq: number, piece: Piece) => {
  const startSquares = getAllNeighbors(fromSq)

  return bfs({
    isAchievable: (sq) => {
      return board.board[sq] === piece
    },
    startSquares,
    getNeighbors: getAllNeighbors,
  })
  return false
}
