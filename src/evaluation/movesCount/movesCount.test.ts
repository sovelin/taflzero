import {beforeAll, describe, expect, it} from "vitest";
import {createBoard, getSquareFromAlgebraic, Piece, setPiece} from "@/board";
import {getPossibleMovesCount} from "./movesCount";
import {initEngine} from "@/engine";

describe('movesCount evaluation', () => {
  beforeAll(() => {
    initEngine()
  })

  it('should be 20 available moves', () => {
    const board = createBoard()
    const sq = getSquareFromAlgebraic('e5')
    setPiece(board, sq, Piece.DEFENDER)

    expect(getPossibleMovesCount(board, sq)).toBe(20)
  })

  it('should be 14 available moves', () => {
    const board = createBoard()
    const sq = getSquareFromAlgebraic('e5')
    setPiece(board, sq, Piece.DEFENDER)
    setPiece(board, getSquareFromAlgebraic('e7'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('c5'), Piece.ATTACKER)

    expect(getPossibleMovesCount(board, sq)).toBe(12)
  })

  it('should be 0 available moves', () => {
    const board = createBoard()
    const sq = getSquareFromAlgebraic('e5')
    setPiece(board, sq, Piece.DEFENDER)
    setPiece(board, getSquareFromAlgebraic('e6'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('e4'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('d5'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('f5'), Piece.ATTACKER)

    expect(getPossibleMovesCount(board, sq)).toBe(0)
  })
})
