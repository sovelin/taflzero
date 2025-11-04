import {describe, it, expect, beforeAll} from "vitest";
import {clearBoard, createBoard, setInitialPosition, setPiece, Piece, Side, HOLE} from "@/board";
import {getSquareFromAlgebraic} from "@/board/utils";
import {getBoardExpects} from "@/board/tests";
import {makeMove} from "./makeMove";
import {createMoveFromAlgebraic} from "../move";
import {printBoard} from "@/board/print";
import {UndoMove} from "@/moves/model/UndoMove";
import {initEngine} from "@/engine";

describe('makeMove', () => {
  beforeAll(() => {
    initEngine()
  })

  const prepare = () => {
    const board = createBoard();
    clearBoard(board);

    return {
      board
    }
  }

  const expectUndoHasCapturedPiece = (undo: UndoMove, sqAlgebraic: string, piece: Piece) => {
    const sq = getSquareFromAlgebraic(sqAlgebraic);
    const captured = undo.captured.find(c => c.sq === sq && c.piece === piece);
    expect(captured).toBeDefined();
  }

  it('simple move', () => {
    const {board} = prepare();
    const {expectSideToBe} = getBoardExpects(board)
    board.sideToMove = Side.ATTACKERS;
    setPiece(board, getSquareFromAlgebraic("a2"), Piece.ATTACKER);
    const move = createMoveFromAlgebraic("a2a4");

    makeMove(board, move);

    const boardExpects = getBoardExpects(board);
    boardExpects.expectAttackerOn("a4");
    boardExpects.expectNoPieceOn("a2");
    boardExpects.expectAttackersCount(1);
    boardExpects.expectDefendersCount(0);
    expectSideToBe(Side.DEFENDERS);
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

  it('should return Undo move structure', () => {
    const FROM = `a2`;
    const TO = `a4`;

    const {board} = prepare();
    board.sideToMove = Side.ATTACKERS;
    setPiece(board, getSquareFromAlgebraic(FROM), Piece.ATTACKER);
    setPiece(board, getSquareFromAlgebraic('a6'), Piece.ATTACKER);
    setPiece(board, getSquareFromAlgebraic('a5'), Piece.DEFENDER);

    const move = createMoveFromAlgebraic([FROM, TO].join(''));
    const undo = makeMove(board, move);
    expect(undo).toBeDefined();
    expect(undo.from).toBe(getSquareFromAlgebraic(FROM));
    expect(undo.to).toBe(getSquareFromAlgebraic(TO));
    expect(undo.captured).toEqual([{
      sq: getSquareFromAlgebraic('a5'),
      piece: Piece.DEFENDER
    }]);
    expect(undo.lastMoveTo).toBe(HOLE);
  })

  describe('shieldwall rule', () => {
    it('if there are 2 rounded pieces on edge, it could be captured', () => {
      const {board} = prepare();
      board.sideToMove = Side.ATTACKERS;
      setPiece(board, getSquareFromAlgebraic("c1"), Piece.ATTACKER);
      setPiece(board, getSquareFromAlgebraic("e2"), Piece.ATTACKER);
      setPiece(board, getSquareFromAlgebraic("f2"), Piece.ATTACKER);
      setPiece(board, getSquareFromAlgebraic("g1"), Piece.ATTACKER);
      setPiece(board, getSquareFromAlgebraic("e1"), Piece.DEFENDER);
      setPiece(board, getSquareFromAlgebraic("f1"), Piece.DEFENDER);
      printBoard(board)

      const move = createMoveFromAlgebraic("c1d1");
      const undo = makeMove(board, move);

      expectUndoHasCapturedPiece(undo, "e1", Piece.DEFENDER);
      expectUndoHasCapturedPiece(undo, "f1", Piece.DEFENDER);
      expect(board.board[getSquareFromAlgebraic("e1")]).toBe(Piece.EMPTY);
      expect(board.board[getSquareFromAlgebraic("f1")]).toBe(Piece.EMPTY);
      expect(undo.captured).toHaveLength(2);
    })

    it('if there are 2 not fully rounded pieces on edge (without e2)-> not captured', () => {
      const {board} = prepare();
      board.sideToMove = Side.ATTACKERS;
      setPiece(board, getSquareFromAlgebraic("c1"), Piece.ATTACKER);
      setPiece(board, getSquareFromAlgebraic("f2"), Piece.ATTACKER);
      setPiece(board, getSquareFromAlgebraic("g1"), Piece.ATTACKER);
      setPiece(board, getSquareFromAlgebraic("e1"), Piece.DEFENDER);
      setPiece(board, getSquareFromAlgebraic("f1"), Piece.DEFENDER);
      printBoard(board)

      const move = createMoveFromAlgebraic("c1d1");
      const undo = makeMove(board, move);

      expect(undo.captured).toHaveLength(0);
      expect(board.board[getSquareFromAlgebraic("e1")]).toBe(Piece.DEFENDER);
      expect(board.board[getSquareFromAlgebraic("f1")]).toBe(Piece.DEFENDER);
    })

    it('one piece surrounded -> no capture', () => {
      const {board} = prepare();
      board.sideToMove = Side.ATTACKERS;
      setPiece(board, getSquareFromAlgebraic("c1"), Piece.ATTACKER);
      setPiece(board, getSquareFromAlgebraic("d3"), Piece.ATTACKER);
      setPiece(board, getSquareFromAlgebraic("e1"), Piece.ATTACKER);
      setPiece(board, getSquareFromAlgebraic("d1"), Piece.DEFENDER);
      printBoard(board)

      const move = createMoveFromAlgebraic("d3d2");
      const undo = makeMove(board, move);

      expect(undo.captured).toHaveLength(0);
      expect(board.board[getSquareFromAlgebraic("d1")]).toBe(Piece.DEFENDER);
    })

    it('if there are 2 rounded pieces on edge, it could be captured (but king not captured)', () => {
      const {board} = prepare();
      board.sideToMove = Side.ATTACKERS;
      setPiece(board, getSquareFromAlgebraic("c1"), Piece.ATTACKER);
      setPiece(board, getSquareFromAlgebraic("e2"), Piece.ATTACKER);
      setPiece(board, getSquareFromAlgebraic("f2"), Piece.ATTACKER);
      setPiece(board, getSquareFromAlgebraic("g1"), Piece.ATTACKER);
      setPiece(board, getSquareFromAlgebraic("e1"), Piece.DEFENDER);
      setPiece(board, getSquareFromAlgebraic("f1"), Piece.KING);
      printBoard(board)

      const move = createMoveFromAlgebraic("c1d1");
      const undo = makeMove(board, move);

      expectUndoHasCapturedPiece(undo, "e1", Piece.DEFENDER);
      expect(board.board[getSquareFromAlgebraic("e1")]).toBe(Piece.EMPTY);
      expect(board.board[getSquareFromAlgebraic("f1")]).toBe(Piece.KING);
      expect(undo.captured).toHaveLength(1);
    })

    it('defenders also could capture', () => {
      const {board} = prepare();
      board.sideToMove = Side.DEFENDERS;
      setPiece(board, getSquareFromAlgebraic("c1"), Piece.DEFENDER);
      setPiece(board, getSquareFromAlgebraic("e2"), Piece.KING);
      setPiece(board, getSquareFromAlgebraic("f2"), Piece.DEFENDER);
      setPiece(board, getSquareFromAlgebraic("g1"), Piece.DEFENDER);
      setPiece(board, getSquareFromAlgebraic("e1"), Piece.ATTACKER);
      setPiece(board, getSquareFromAlgebraic("f1"), Piece.ATTACKER);
      printBoard(board)

      const move = createMoveFromAlgebraic("c1d1");
      const undo = makeMove(board, move);

      expectUndoHasCapturedPiece(undo, "e1", Piece.ATTACKER);
      expectUndoHasCapturedPiece(undo, "f1", Piece.ATTACKER);
      expect(board.board[getSquareFromAlgebraic("e1")]).toBe(Piece.EMPTY);
      expect(board.board[getSquareFromAlgebraic("f1")]).toBe(Piece.EMPTY);
      expect(undo.captured).toHaveLength(2);
    })

    it('capture on corner with shieldwall', () => {
      const {board} = prepare();
      board.sideToMove = Side.DEFENDERS;
      setPiece(board, getSquareFromAlgebraic("b2"), Piece.DEFENDER);
      setPiece(board, getSquareFromAlgebraic("c2"), Piece.DEFENDER);
      setPiece(board, getSquareFromAlgebraic("e1"), Piece.DEFENDER);
      setPiece(board, getSquareFromAlgebraic("b1"), Piece.ATTACKER);
      setPiece(board, getSquareFromAlgebraic("c1"), Piece.ATTACKER);
      printBoard(board)

      const move = createMoveFromAlgebraic("e1d1");
      const undo = makeMove(board, move);

      expectUndoHasCapturedPiece(undo, "b1", Piece.ATTACKER);
      expectUndoHasCapturedPiece(undo, "c1", Piece.ATTACKER);
      expect(board.board[getSquareFromAlgebraic("b1")]).toBe(Piece.EMPTY);
      expect(board.board[getSquareFromAlgebraic("c1")]).toBe(Piece.EMPTY);
      expect(undo.captured).toHaveLength(2);
    })

    it('standard captures should works too. more complex test', () => {
      const {board} = prepare();
      board.sideToMove = Side.ATTACKERS;
      setPiece(board, getSquareFromAlgebraic("b2"), Piece.ATTACKER);
      setPiece(board, getSquareFromAlgebraic("c2"), Piece.ATTACKER);
      setPiece(board, getSquareFromAlgebraic("f2"), Piece.ATTACKER);
      setPiece(board, getSquareFromAlgebraic("e1"), Piece.ATTACKER);
      setPiece(board, getSquareFromAlgebraic("d1"), Piece.DEFENDER);
      setPiece(board, getSquareFromAlgebraic("b1"), Piece.DEFENDER);
      setPiece(board, getSquareFromAlgebraic("c1"), Piece.KING);
      setPiece(board, getSquareFromAlgebraic("d3"), Piece.DEFENDER);
      setPiece(board, getSquareFromAlgebraic("d4"), Piece.ATTACKER);
      printBoard(board)

      const move = createMoveFromAlgebraic("f2d2");
      const undo = makeMove(board, move);

      expectUndoHasCapturedPiece(undo, "b1", Piece.DEFENDER);
      expectUndoHasCapturedPiece(undo, "d1", Piece.DEFENDER);
      expectUndoHasCapturedPiece(undo, "d3", Piece.DEFENDER);
      expect(board.board[getSquareFromAlgebraic("b1")]).toBe(Piece.EMPTY);
      expect(board.board[getSquareFromAlgebraic("d1")]).toBe(Piece.EMPTY);
      expect(board.board[getSquareFromAlgebraic("d3")]).toBe(Piece.EMPTY);
      expect(undo.captured).toHaveLength(3);
    })
  })
})
