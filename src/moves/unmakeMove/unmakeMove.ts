import {Board, clearPiece, setPiece} from "@/board";
import {UndoMove} from "@/moves/model/UndoMove";
import {flipSide} from "@/board/board";

function removePositionFromRepTable(board: Board) {
  if (board.repTable.has(board.zobrist)) {
    board.repTable.set(board.zobrist, board.repTable.get(board.zobrist)! - 1);

    if (board.repTable.get(board.zobrist)! <= 0) {
      board.repTable.delete(board.zobrist);
    }
  }
}

export function unmakeMove(board: Board, undo: UndoMove) {
  removePositionFromRepTable(board);
  clearPiece(board, undo.to);
  setPiece(board, undo.from, undo.movedPiece);

  for (const captured of undo.captured) {
    setPiece(board, captured.sq, captured.piece);
  }

  flipSide(board);
  board.lastMoveTo = undo.lastMoveTo;
}
