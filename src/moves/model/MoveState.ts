import {Piece, Side} from "@/board";

export interface MoveState {
  from: number;
  to: number;
  captured: { sq: number; piece: Piece }[];
  zobristBefore: bigint;
  rowOccBefore: Uint16Array;
  colOccBefore: Uint16Array;
  sideBefore: Side;
}
