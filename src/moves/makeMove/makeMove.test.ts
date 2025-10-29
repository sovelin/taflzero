import {describe, it, expect} from "vitest";
import {clearBoard, createBoard, setInitialPosition, setPiece, Piece, Side} from "@/board";
import {getSquareFromAlgebraic} from "@/board/utils";
import {getBoardExpects} from "@/board/tests";
import {makeMove} from "./makeMove";
import {createMoveFromAlgebraic} from "../move";

describe('makeMove', () => {
  const prepare = () => {
    const board = createBoard();
    clearBoard(board);

    return {
      board
    }
  }

  it('simple move', () => {
    const {board} = prepare();
    board.sideToMove = Side.ATTACKERS;
    setPiece(board, getSquareFromAlgebraic("a2"), Piece.ATTACKER);
    const move = createMoveFromAlgebraic("a2a4");

    makeMove(board, move);

    const boardExpects = getBoardExpects(board);
    boardExpects.expectAttackerOn("a4");
    boardExpects.expectNoPieceOn("a2");
    boardExpects.expectAttackersCount(1);
    boardExpects.expectDefendersCount(0);
  })

  it('capture defender in sandwich', () => {
    const {board} = prepare();
    board.sideToMove = Side.ATTACKERS;
    setPiece(board, getSquareFromAlgebraic("a2"), Piece.ATTACKER);
    setPiece(board, getSquareFromAlgebraic("a4"), Piece.DEFENDER);
    setPiece(board, getSquareFromAlgebraic("a5"), Piece.ATTACKER);
    const move = createMoveFromAlgebraic("a2a3");

    makeMove(board, move);

    const boardExpects = getBoardExpects(board);
    boardExpects.expectAttackerOn("a3");
    boardExpects.expectAttackerOn("a5");
    boardExpects.expectNoPieceOn("a4");
    boardExpects.expectAttackersCount(2);
    boardExpects.expectDefendersCount(0);
  })

  it('king should not be captured in sandwich', () => {
    const {board} = prepare();
    board.sideToMove = Side.ATTACKERS;
    setPiece(board, getSquareFromAlgebraic("a2"), Piece.ATTACKER);
    setPiece(board, getSquareFromAlgebraic("a4"), Piece.KING);
    setPiece(board, getSquareFromAlgebraic("a5"), Piece.ATTACKER);
    const move = createMoveFromAlgebraic("a2a3");

    makeMove(board, move);

    const boardExpects = getBoardExpects(board);
    boardExpects.expectAttackerOn("a3");
    boardExpects.expectAttackerOn("a5");
    boardExpects.expectKingOn("a4");
    boardExpects.expectAttackersCount(2);
    boardExpects.expectDefendersCount(0);
  })

  it('make any move on initial position, when board has all pieces, to check edge caase with attackers/defenders overflow', () => {
    const {board} = prepare();
    setInitialPosition(board);
    board.sideToMove = Side.ATTACKERS;
    const move = createMoveFromAlgebraic("d1d2");

    expect(() => makeMove(board, move)).not.toThrowError();
  })
})
