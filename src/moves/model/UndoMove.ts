import {Piece} from "@/board";

export interface CapturedPiece {
  sq: number;
  piece: Piece;
}

export interface UndoMove {
  from: number;
  to: number;
  captured: CapturedPiece[];
  movedPiece: Piece;
  lastMoveTo: number;
}
