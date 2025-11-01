import {describe, it, expect} from 'vitest';
import {createBoard, setInitialPosition, Side} from "@/board";
import {makeMove} from "@/moves";
import {printBoard} from "@/board/print";
import {checkTerminal} from "./terminal";
import {createMoveFromAlgebraic} from "@/moves/move";

describe('Terminal Evaluation Tests', () => {
  it('3-repeatitions is always win for attackers', () => {
    const board = createBoard()
    setInitialPosition(board)
    printBoard(board)
    makeMove(board, createMoveFromAlgebraic('a4b4'))
    makeMove(board, createMoveFromAlgebraic('e5d5'))
    makeMove(board, createMoveFromAlgebraic('b4a4'))
    makeMove(board, createMoveFromAlgebraic('d5e5'))
    makeMove(board, createMoveFromAlgebraic('a4b4'))
    makeMove(board, createMoveFromAlgebraic('e5d5'))
    makeMove(board, createMoveFromAlgebraic('b4a4'))
    makeMove(board, createMoveFromAlgebraic('d5e5'))
    makeMove(board, createMoveFromAlgebraic('a4b4'))

    const isTerminal = checkTerminal(board)

    expect(isTerminal).toBe(Side.ATTACKERS)
  })

  it('2-repeatitions is not terminated', () => {
    const board = createBoard()
    setInitialPosition(board)
    printBoard(board)
    makeMove(board, createMoveFromAlgebraic('a4b4'))
    makeMove(board, createMoveFromAlgebraic('e5d5'))
    makeMove(board, createMoveFromAlgebraic('b4a4'))
    makeMove(board, createMoveFromAlgebraic('d5e5'))
    makeMove(board, createMoveFromAlgebraic('a4b4'))
    makeMove(board, createMoveFromAlgebraic('e5d5'))
    makeMove(board, createMoveFromAlgebraic('b4a4'))

    const isTerminal = checkTerminal(board)

    expect(isTerminal).toBeNull()
  })
})
