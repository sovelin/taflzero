import {Board, clearPiece, getOppositeSide, setPiece, Side} from "@/board";
import {UndoMove} from "@/moves/model/UndoMove";

export function unmakeMove(board: Board, undo: UndoMove) {
  clearPiece(board, undo.to);
  setPiece(board, undo.from, undo.movedPiece);

  for (const captured of undo.captured) {
    setPiece(board, captured.sq, captured.piece);
  }

  board.sideToMove = getOppositeSide(board.sideToMove);
}
