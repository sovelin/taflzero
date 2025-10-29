import {expect} from "vitest";
import {Piece} from "../types";
import {Board} from "../model/Board";
import {getSquareFromAlgebraic} from "../utils";

export const getBoardExpects = (board: Board) => {
  return {
    expectAttackerOn: (sqStr: string) => {
      const sq = getSquareFromAlgebraic(sqStr);
      expect(board.board[sq]).toBe(Piece.ATTACKER);
      expect(board.attackers.includes(sq)).toBeTruthy()
    },
    expectDefenderOn: (sqStr: string) => {
      const sq = getSquareFromAlgebraic(sqStr);
      expect(board.board[sq]).toBe(Piece.DEFENDER);
      expect(board.defenders.includes(sq)).toBeTruthy()
    },
    expectKingOn: (sqStr: string) => {
      const sq = getSquareFromAlgebraic(sqStr);
      expect(board.board[sq]).toBe(Piece.KING);
      expect(board.kingSq).toBe(sq);
    },
    expectAttackersCount: (count: number) => {
      expect(board.attackersCount).toBe(count);
    },
    expectDefendersCount: (count: number) => {
      expect(board.defendersCount).toBe(count);
    },
    expectNoPieceOn: (sqStr: string) => {
      const sq = getSquareFromAlgebraic(sqStr);
      expect(board.board[sq]).toBe(Piece.EMPTY);
      expect(!board.attackers.includes(sq)).toBeTruthy()
      expect(!board.defenders.includes(sq)).toBeTruthy()
      expect(board.kingSq).not.toBe(sq);
    },
    expectKingNotExists: () => {
      expect(board.kingSq).toBe(-1);
    },
  }
}
