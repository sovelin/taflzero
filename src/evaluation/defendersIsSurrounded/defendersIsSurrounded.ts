import {Board, getCornersSq, Piece} from "@/board";
import {bfs} from "@/utils";
import {isEdgeSquare} from "@/board/utils";

const corners = getCornersSq()

export const defendersIsSurrounded = (board: Board) => {
  const visited = bfs({
    isAchievable: (sq) => board.board[sq] !== Piece.ATTACKER,
    startSquares: corners,
  })

  for(const sq of [...visited]) {
    if(board.board[sq] === Piece.DEFENDER || board.board[sq] === Piece.KING) {
      return false;
    }
  }

  const innerSquares = bfs({
    isAchievable: (sq) => board.board[sq] !== Piece.ATTACKER,
    startSquares: [board.kingSq],
  })

  for(const sq of [...innerSquares]) {


    if (isEdgeSquare(sq)) {
      return false;
    }
  }

  return true;
}
