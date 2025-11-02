import {Board, getCornersSq, getSideByPiece, getThroneSq, Piece} from "@/board";

const enemySquares = [...getCornersSq(), getThroneSq()]

export const isCapturePossible = (
  board: Board,
  targetSq: number,
  enemySq1: number,
  enemySq2: number
) => {
  const attackerPiece1 = board.board[enemySq1];
  const attackerPiece2 = board.board[enemySq2];

  const attackerPiece = attackerPiece1 || attackerPiece2;
  const targetPieceForAttackers = board.board[targetSq];
  console.log({attackerPiece1, attackerPiece2})


  if (attackerPiece1
    && attackerPiece2
    && ((attackerPiece1 !== attackerPiece2)
      && !(attackerPiece1 === Piece.KING && attackerPiece2 === Piece.DEFENDER)
      && !(attackerPiece2 === Piece.KING && attackerPiece1 === Piece.DEFENDER)
    )
  ) {
    return false;
  }


  if (targetPieceForAttackers === Piece.EMPTY || targetPieceForAttackers === Piece.KING) {
    return false;
  }

  const attackerSide = getSideByPiece(attackerPiece)
  const targetPieceSide = getSideByPiece(targetPieceForAttackers);
  // no corners -> simple check
  if (attackerPiece1 && attackerPiece2) {
    return attackerSide !== targetPieceSide;
  }



  if (!targetPieceForAttackers || targetPieceForAttackers === Piece.KING) {
    return false;
  }

  if (enemySquares.includes(enemySq1)) {
    return attackerSide !== targetPieceSide
  }

  if (enemySquares.includes(enemySq2)) {
    return attackerSide !== targetPieceSide
  }

  return  false
}
