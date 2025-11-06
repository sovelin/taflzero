import {describe, it, expect} from 'vitest';
import {createBoard, setInitialPosition, setPiece} from "../board";
import {getFEN, setFEN} from "./fen";
import {getBoardExpects} from "../tests/index";
import {Piece, Side} from "../types";
import {printBoard} from "@/board/print";
import {getSquareFromAlgebraic, HOLE} from "@/board";

describe('FEN Parser', () => {
  describe('setFEN', () => {
    it('one king on throne, 2 attackers and 2 defenders, defenders to move', () => {
      const fen = '11/4a6/11/11/7d3/5k5/11/1d9/8a2/11/10a d'

      const board = createBoard()
      const boardExpects = getBoardExpects(board)
      setFEN(board, fen)

      boardExpects.expectAttackersCount(3)
      boardExpects.expectDefendersCount(2)
      printBoard(board)
      boardExpects.expectAttackerOn('i3')
      boardExpects.expectAttackerOn('e10')
      boardExpects.expectKingOn('f6')
      boardExpects.expectDefenderOn('b4')
      boardExpects.expectDefenderOn('h7')
      boardExpects.expectAttackerOn('k1')

      expect(board.sideToMove).toBe(Side.DEFENDERS)
    })

    it('attackers to move', () => {
      const fen = '11/4a6/11/11/7d3/5k5/11/1d9/8a2/11/11 a'

      const board = createBoard()

      setFEN(board, fen)

      expect(board.sideToMove).toBe(Side.ATTACKERS)
    })

    it('set twice', () => {
      const fen = '11/4a6/11/11/7d3/5k5/11/1d9/8a2/11/11 a'

      const board = createBoard()

      setFEN(board, '11/4a6/11/11/7d3/5k5/11/1d9/8a2/11/11 a')
      setFEN(board, '11/11/11/11/11/11/11/11/11/11/11 a')

      expect(board.sideToMove).toBe(Side.ATTACKERS)
      expect(board.attackersCount).toBe(0)
      expect(board.defendersCount).toBe(0)
      expect(board.kingSq).toBe(HOLE)
    })
  })

  describe('getFEN', () => {
    it('one king on throne, 2 attackers and 2 defenders, attackers to move', () => {
      const board = createBoard()
      setPiece(board, getSquareFromAlgebraic('i3'), Piece.ATTACKER)
      setPiece(board, getSquareFromAlgebraic('f6'), Piece.KING)
      setPiece(board, getSquareFromAlgebraic('b4'), Piece.DEFENDER)
      setPiece(board, getSquareFromAlgebraic('h7'), Piece.DEFENDER)
      setPiece(board, getSquareFromAlgebraic('e10'), Piece.ATTACKER)
      setPiece(board, getSquareFromAlgebraic('k1'), Piece.ATTACKER)

      printBoard(board)

      const fen = getFEN(board)

      expect(fen).toBe('11/4a6/11/11/7d3/5k5/11/1d9/8a2/11/10a a')
    })
  });

  it('zobrists should be same', () => {
    const board = createBoard()
    setInitialPosition(board)
    const initialFen = getFEN(board)
    const initialZobrist = board.zobrist

    setFEN(board, initialFen)
    const newZobrist = board.zobrist

    expect(initialZobrist).toBe(newZobrist)
  })
})
