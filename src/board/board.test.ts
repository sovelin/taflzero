import {describe, expect, it} from 'vitest';
import {clearBoard, clearPiece, createBoard, setInitialPosition, setPiece} from "./board";
import {getSquareFromAlgebraic} from "./utils";
import {Piece} from "./types";
import {getBoardExpects} from "./tests";
import {getBinary} from "@/utils/utils";

describe('Board Tests', () => {
  const prepare = () => {
    const board = createBoard();
    clearBoard(board);

    return {
      board,
      ...getBoardExpects(board)
    }
  }

  it('correctly set piece', () => {
    const {board, expectAttackerOn, expectAttackersCount, expectDefendersCount} = prepare()

    // write all test
    setPiece(board, getSquareFromAlgebraic("b1"), Piece.ATTACKER);
    expect(getBinary(board.rowOcc[0])).toBe('01000000000');
    expect(getBinary(board.colOcc[1])).toBe('10000000000');
    expectAttackerOn("b1");
    expectAttackersCount(1);
    expectDefendersCount(0);
  })

  it('correctly set 2 pieces', () => {
    const {board, expectAttackerOn, expectAttackersCount, expectDefenderOn, expectDefendersCount} = prepare()

    setPiece(board, getSquareFromAlgebraic("b1"), Piece.ATTACKER);
    setPiece(board, getSquareFromAlgebraic("c2"), Piece.DEFENDER);

    expect(getBinary(board.rowOcc[0])).toBe('01000000000');
    expect(getBinary(board.colOcc[1])).toBe('10000000000');
    expect(getBinary(board.rowOcc[1])).toBe('00100000000');
    expect(getBinary(board.colOcc[2])).toBe('01000000000');
    expectAttackerOn("b1");
    expectDefenderOn("c2");
    expectAttackersCount(1);
    expectDefendersCount(1);
  })

  it('correctly set and clear pieces', () => {
    const {board, expectAttackerOn, expectAttackersCount, expectDefenderOn, expectDefendersCount} = prepare()

    setPiece(board, getSquareFromAlgebraic("b1"), Piece.ATTACKER);
    setPiece(board, getSquareFromAlgebraic("c2"), Piece.DEFENDER);
    clearPiece(board, getSquareFromAlgebraic("c2"));

    expect(getBinary(board.rowOcc[0])).toBe('01000000000');
    expect(getBinary(board.colOcc[1])).toBe('10000000000');
    expectAttackerOn("b1");
    expectAttackersCount(1);
    expectDefendersCount(0);
  })

  it('correctly set king', () => {
    const {board, expectKingOn} = prepare()
    setPiece(board, getSquareFromAlgebraic("f6"), Piece.KING);
    expect(getBinary(board.rowOcc[5])).toBe('00000100000');
    expect(getBinary(board.colOcc[5])).toBe('00000100000');
    expectKingOn("f6");
  })

  it('correctly clear king', () => {
    const {board, expectNoPieceOn, expectKingNotExists} = prepare()
    setPiece(board, getSquareFromAlgebraic("f6"), Piece.KING);
    clearPiece(board, getSquareFromAlgebraic("f6"));

    expectNoPieceOn(("f6"));
    expectKingNotExists()
  })

  it('add and clear defender', () => {
    const {board, expectNoPieceOn} = prepare()
    setPiece(board, getSquareFromAlgebraic("a4"), Piece.DEFENDER);
    clearPiece(board, getSquareFromAlgebraic("a4"));

    expectNoPieceOn(("a4"));
  })

  it('add and clear attacker', () => {
    const {board, expectNoPieceOn} = prepare()
    setPiece(board, getSquareFromAlgebraic("a4"), Piece.ATTACKER);
    clearPiece(board, getSquareFromAlgebraic("a4"));

    expectNoPieceOn(("a4"));
  })

  it('correct attacker indicies', () => {
    const {board} = prepare()
    setPiece(board, getSquareFromAlgebraic("a1"), Piece.ATTACKER);
    setPiece(board, getSquareFromAlgebraic("b2"), Piece.ATTACKER);
    setPiece(board, getSquareFromAlgebraic("c3"), Piece.ATTACKER);
    clearPiece(board, getSquareFromAlgebraic("a1"));

    expect(board.pieceIndexBySquare[getSquareFromAlgebraic("b2")]).toBe(1);
    expect(board.pieceIndexBySquare[getSquareFromAlgebraic("c3")]).toBe(0);
    expect(board.pieceIndexBySquare[getSquareFromAlgebraic("a1")]).toBe(-1);
  })

  it('correct defender indicies', () => {
    const {board} = prepare()
    setPiece(board, getSquareFromAlgebraic("a1"), Piece.DEFENDER);
    setPiece(board, getSquareFromAlgebraic("b2"), Piece.DEFENDER);
    setPiece(board, getSquareFromAlgebraic("c3"), Piece.DEFENDER);
    clearPiece(board, getSquareFromAlgebraic("a1"));

    expect(board.pieceIndexBySquare[getSquareFromAlgebraic("b2")]).toBe(1);
    expect(board.pieceIndexBySquare[getSquareFromAlgebraic("c3")]).toBe(0);
    expect(board.pieceIndexBySquare[getSquareFromAlgebraic("a1")]).toBe(-1);
  })

  it('if board is full, should be impossible to add more pieces', () => {
    const {board} = prepare()
    setInitialPosition(board)

    expect(() => setPiece(board, getSquareFromAlgebraic("d4"), Piece.ATTACKER)).toThrowError();
    expect(() => setPiece(board, getSquareFromAlgebraic("d4"), Piece.DEFENDER)).toThrowError();
  })
})
