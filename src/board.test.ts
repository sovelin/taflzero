import {describe, expect, it} from 'vitest';
import {clearBoard, clearPiece, createBoard, setPiece} from "./board";
import {getBinary, getSquareFromAlgebraic} from "./utils";
import {Piece} from "./types";

describe('Board Tests', () => {
  const prepare = () => {
    const board = createBoard();
    clearBoard(board);

    return {
      board,
      expectAttackerOn: (sq: number) => {
        expect(board.board[sq]).toBe(Piece.ATTACKER);
        expect(board.attackers.includes(sq)).toBeTruthy()
      },
      expectDefenderOn: (sq: number) => {
        expect(board.board[sq]).toBe(Piece.DEFENDER);
        expect(board.defenders.includes(sq)).toBeTruthy()
      },
      expectKingOn: (sq: number) => {
        expect(board.board[sq]).toBe(Piece.KING);
        expect(board.kingSq).toBe(sq);
      },
      expectAttackersCount: (count: number) => {
        expect(board.attackersCount).toBe(count);
      },
      expectDefendersCount: (count: number) => {
        expect(board.defendersCount).toBe(count);
      },
      expectNoPieceOn: (sq: number) => {
        expect(board.board[sq]).toBe(Piece.EMPTY);
        expect(!board.attackers.includes(sq)).toBeTruthy()
        expect(!board.defenders.includes(sq)).toBeTruthy()
        expect(board.kingSq).not.toBe(sq);
      },
      expectKingNotExists: () => {
        expect(board.kingSq).toBe(-1);
      }
    }
  }

  it('correctly set piece', () => {
    const {board, expectAttackerOn, expectAttackersCount, expectDefendersCount} = prepare()

    // write all test
    setPiece(board, getSquareFromAlgebraic("b1"), Piece.ATTACKER);
    expect(getBinary(board.rowOcc[0])).toBe('01000000000');
    expect(getBinary(board.colOcc[1])).toBe('10000000000');
    expectAttackerOn(getSquareFromAlgebraic("b1"));
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
    expectAttackerOn(getSquareFromAlgebraic("b1"));
    expectDefenderOn(getSquareFromAlgebraic("c2"));
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
    expectAttackerOn(getSquareFromAlgebraic("b1"));
    expectAttackersCount(1);
    expectDefendersCount(0);
  })

  it('correctly set king', () => {
    const {board, expectKingOn} = prepare()
    setPiece(board, getSquareFromAlgebraic("f6"), Piece.KING);
    expect(getBinary(board.rowOcc[5])).toBe('00000100000');
    expect(getBinary(board.colOcc[5])).toBe('00000100000');
    expectKingOn(getSquareFromAlgebraic("f6"));
  })

  it('correctly clear king', () => {
    const {board, expectNoPieceOn, expectKingNotExists} = prepare()
    setPiece(board, getSquareFromAlgebraic("f6"), Piece.KING);
    clearPiece(board, getSquareFromAlgebraic("f6"));

    expectNoPieceOn(getSquareFromAlgebraic("f6"));
    expectKingNotExists()
  })
})
