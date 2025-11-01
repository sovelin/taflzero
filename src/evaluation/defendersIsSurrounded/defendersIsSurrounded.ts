import {Board, getCornersSq, HOLE, Piece, SQS} from "@/board";
import {bfs} from "@/utils";
import {getAllNeighbors, isEdgeSquare} from "@/board/utils";

const corners = getCornersSq()

const isCalculateNeeded = (board: Board) => {
  const lastMoveSq = board.lastMoveTo;

  if (lastMoveSq === HOLE) {
    // no moves played yet -> calculation needed
    return true;
  }

  if (board.board[lastMoveSq] !== Piece.ATTACKER) {
    // defender moved -> calculation not needed
    return false;
  }

  const allSiblings = getAllNeighbors(lastMoveSq)

  let attackerNearbyCount = 0;

  for (let i = 0; i < allSiblings.length; i++) {
    const sq = allSiblings[i];

    if (board.board[sq] === Piece.ATTACKER) {
      attackerNearbyCount++;

      // calculate needed if 2 or more attackers are nearby
      if (attackerNearbyCount >= 2) {
        return true;
      }
    }
  }

  // cycle detection check

  return false
}

export const defendersIsSurrounded = (board: Board) => {
  if (!isCalculateNeeded(board)) {
    return false;
  }


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
