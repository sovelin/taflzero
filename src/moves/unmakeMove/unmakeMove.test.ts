import {describe, expect, it} from "vitest";
import {clearBoard, createBoard, getSquareFromAlgebraic, Piece, setPiece, Side} from "@/board";
import {printBoard} from "@/board/print";
import {getBoardExpects} from "@/board/tests";
import {UndoMove} from "../model/UndoMove";
import {unmakeMove} from "@/moves";

describe('Unmake Move Tests', () => {
  it('should move and move back a piece correctly', () => {
    const board = createBoard();
    const {expectAttackerOn, expectDefenderOn, expectAttackersCount, expectDefendersCount, expectSideToBe} = getBoardExpects(board)
    clearBoard(board);
    board.sideToMove = Side.DEFENDERS;
    setPiece(board, getSquareFromAlgebraic("b5"), Piece.ATTACKER);
    setPiece(board, getSquareFromAlgebraic("b7"), Piece.ATTACKER);

    const undo: UndoMove = {
      from: getSquareFromAlgebraic("b4"),
      to: getSquareFromAlgebraic("b5"),
      captured: [{
        sq: getSquareFromAlgebraic("b6"),
        piece: Piece.DEFENDER
      }],
      movedPiece: Piece.ATTACKER,
      lastMoveTo: 10
    }

    unmakeMove(board, undo);
    printBoard(board)
    expectAttackerOn("b4");
    expectAttackerOn("b7");
    expectAttackersCount(2);

    expectDefenderOn("b6");
    expectDefendersCount(1);
    expectSideToBe(Side.ATTACKERS);
    expect(board.lastMoveTo).toBe(10);

    printBoard(board)
  })
})
