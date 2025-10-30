import {it, describe, expect} from 'vitest';
import {createBoard, getSquareFromAlgebraic, Piece, setPiece, SQS} from "@/board";
import {bfs, } from "../bfs";

describe('BFS Tests', () => {
  it('empty board, all achievable', () => {
    const board = createBoard()

    const res = bfs(board, {
      isAchievable: () => true,
      startSquares: [0],
    })

    expect(res.size).toBe(SQS)
  })

  it('go from corners and restricted by area', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic('d4'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('d3'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('d2'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('d1'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('c4'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('b4'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('a4'), Piece.ATTACKER)

    const res = bfs(board, {
      isAchievable: (sq) => board.board[sq] === Piece.EMPTY,
      startSquares: [getSquareFromAlgebraic('a1')],
    })

    expect(res.size).toBe(9)
    expect(res.has(getSquareFromAlgebraic('a1'))).toBe(true)
    expect(res.has(getSquareFromAlgebraic('a2'))).toBe(true)
    expect(res.has(getSquareFromAlgebraic('a3'))).toBe(true)
    expect(res.has(getSquareFromAlgebraic('b1'))).toBe(true)
    expect(res.has(getSquareFromAlgebraic('b2'))).toBe(true)
    expect(res.has(getSquareFromAlgebraic('b3'))).toBe(true)
    expect(res.has(getSquareFromAlgebraic('c1'))).toBe(true)
    expect(res.has(getSquareFromAlgebraic('c2'))).toBe(true)
    expect(res.has(getSquareFromAlgebraic('c3'))).toBe(true)
  })

  it('go from different corners, one corner restricted by area', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic('d4'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('d3'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('d2'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('d1'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('c4'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('b4'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('a4'), Piece.ATTACKER)

    const res = bfs(board, {
      isAchievable: (sq) => board.board[sq] === Piece.EMPTY,
      startSquares: [
        getSquareFromAlgebraic('a1'),
        getSquareFromAlgebraic('k1'),
      ],
    })

    expect(res.size).toBe(SQS - 7)
  })
})
