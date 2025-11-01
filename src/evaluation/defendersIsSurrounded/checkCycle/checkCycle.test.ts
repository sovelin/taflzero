import {describe, expect, it} from "vitest";
import {createBoard, getSquareFromAlgebraic, Piece, setPiece} from "@/board";
import {printBoard} from "@/board/print";
import {checkCycle} from "@/evaluation/defendersIsSurrounded/checkCycle/checkCycle";

describe('hasCycle', () => {
  it('have simple cycle (4 attackers)', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic("f4"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("g5"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("h4"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("g3"), Piece.ATTACKER)
    printBoard(board)

    const hasCycle = checkCycle(board, getSquareFromAlgebraic("f4"), Piece.ATTACKER)
    expect(hasCycle).toBeTruthy();
  })

  it('dont have cycle if starting from empty', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic("f4"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("g5"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("h4"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("g3"), Piece.ATTACKER)
    printBoard(board)

    const hasCycle = checkCycle(board, getSquareFromAlgebraic("c2"), Piece.ATTACKER)
    expect(hasCycle).toBeFalsy();
  })

  it('dont have cycle if different piece in cycle', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic("f4"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("g5"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("h4"), Piece.DEFENDER)
    setPiece(board, getSquareFromAlgebraic("g3"), Piece.ATTACKER)
    printBoard(board)

    const hasCycle = checkCycle(board, getSquareFromAlgebraic("f4"), Piece.ATTACKER)
    expect(hasCycle).toBeFalsy();
  })

  it('dont have cycle if only 2 pieces', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic("f4"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("g5"), Piece.ATTACKER)
    printBoard(board)

    const hasCycle = checkCycle(board, getSquareFromAlgebraic("f4"), Piece.ATTACKER)
    expect(hasCycle).toBeFalsy();
  })
})
