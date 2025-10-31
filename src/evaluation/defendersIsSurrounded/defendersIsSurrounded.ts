import {Board, getCornersSq, Piece, SQS} from "@/board";
import {bfs} from "@/utils";
import {isEdgeSquare} from "@/board/utils";

const corners = getCornersSq()

export const defendersIsSurrounded = (board: Board) => {
  const visited = bfs({
    isAchievable: (sq) => board.board[sq] !== Piece.ATTACKER,
    startSquares: corners,
  })

  for(let i = 0; i < SQS; i++) {
    if (!visited[i]) {
      continue;
    }

    if(board.board[i] === Piece.DEFENDER || board.board[i] === Piece.KING) {
      return false;
    }
  }

  const innerSquares = bfs({
    isAchievable: (sq) => board.board[sq] !== Piece.ATTACKER,
    startSquares: [board.kingSq],
  })

  for(let i = 0; i < SQS; i++) {
    if (innerSquares[i] && isEdgeSquare(i)) {
      return false;
    }
  }

  return true;
}
