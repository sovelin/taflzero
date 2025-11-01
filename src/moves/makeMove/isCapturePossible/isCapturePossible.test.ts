import {describe, expect, it} from "vitest";
import {isCapturePossible} from "./isCapturePossible";
import {createBoard, getSquareFromAlgebraic, getThroneSq, Piece, setPiece} from "@/board";

describe('isCapturePossible', () => {
  it('attacker capture defender in the simple cell', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic("c2"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("c4"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("c3"), Piece.DEFENDER)
    const isPossible = isCapturePossible(
      board,
      getSquareFromAlgebraic("c3"),
      getSquareFromAlgebraic("c2"),
      getSquareFromAlgebraic("c4")
    )
    expect(isPossible).toBe(true)
  })

  it('defender capture attacker in the simple cell', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic("c2"), Piece.DEFENDER)
    setPiece(board, getSquareFromAlgebraic("c4"), Piece.DEFENDER)
    setPiece(board, getSquareFromAlgebraic("c3"), Piece.ATTACKER)

    const isPossible = isCapturePossible(
      board,
      getSquareFromAlgebraic("c3"),
      getSquareFromAlgebraic("c2"),
      getSquareFromAlgebraic("c4")
    )
    expect(isPossible).toBe(true)
  })



  it('attacker cannot capture king', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic("c2"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("c4"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("c3"), Piece.KING)

    const isPossible = isCapturePossible(
      board,
      getSquareFromAlgebraic("c3"),
      getSquareFromAlgebraic("c2"),
      getSquareFromAlgebraic("c4")
    )
    expect(isPossible).toBe(false)
  })

  describe('captures on the corners', () => {
    it('king can eat attackers on corners', () => {
      const board = createBoard()
      setPiece(board, getSquareFromAlgebraic("a2"), Piece.ATTACKER)
      setPiece(board, getSquareFromAlgebraic("a3"), Piece.KING)

      const isPossible = isCapturePossible(
        board,
        getSquareFromAlgebraic("a2"),
        getSquareFromAlgebraic("a1"),
        getSquareFromAlgebraic("a3")
      )
      expect(isPossible).toBe(true)
    })

    it('king can eat attackers on corners, swap params', () => {
      const board = createBoard()
      setPiece(board, getSquareFromAlgebraic("a2"), Piece.ATTACKER)
      setPiece(board, getSquareFromAlgebraic("a3"), Piece.KING)

      const isPossible = isCapturePossible(
        board,
        getSquareFromAlgebraic("a2"),
        getSquareFromAlgebraic("a3"),
        getSquareFromAlgebraic("a1"),
      )
      expect(isPossible).toBe(true)
    })

    it('defender can eat attackers on corners', () => {
      const board = createBoard()
      setPiece(board, getSquareFromAlgebraic("a2"), Piece.ATTACKER)
      setPiece(board, getSquareFromAlgebraic("a3"), Piece.DEFENDER)

      const isPossible = isCapturePossible(
        board,
        getSquareFromAlgebraic("a2"),
        getSquareFromAlgebraic("a1"),
        getSquareFromAlgebraic("a3")
      )
      expect(isPossible).toBe(true)
    })

    it('attacker can eat defender on corners', () => {
      const board = createBoard()
      setPiece(board, getSquareFromAlgebraic("a2"), Piece.DEFENDER)
      setPiece(board, getSquareFromAlgebraic("a3"), Piece.ATTACKER)

      const isPossible = isCapturePossible(
        board,
        getSquareFromAlgebraic("a2"),
        getSquareFromAlgebraic("a1"),
        getSquareFromAlgebraic("a3")
      )
      expect(isPossible).toBe(true)
    })

    it('attacker can eat defender on corners, other corner', () => {
      const board = createBoard()
      setPiece(board, getSquareFromAlgebraic("k10"), Piece.DEFENDER)
      setPiece(board, getSquareFromAlgebraic("k9"), Piece.ATTACKER)

      const isPossible = isCapturePossible(
        board,
        getSquareFromAlgebraic("k10"),
        getSquareFromAlgebraic("k11"),
        getSquareFromAlgebraic("k9")
      )
      expect(isPossible).toBe(true)
    })
  })

  describe('captures near the throne', () => {
    it('king can eat attacker near the throne', () => {
      const board = createBoard()
      setPiece(board, getSquareFromAlgebraic("f5"), Piece.ATTACKER)
      setPiece(board, getSquareFromAlgebraic("f4"), Piece.KING)

      const isPossible = isCapturePossible(
        board,
        getSquareFromAlgebraic("f5"),
        getSquareFromAlgebraic("f6"),
        getSquareFromAlgebraic("f4")
      )
      expect(isPossible).toBe(true)
    })

    it('attacker can eat defender near the throne', () => {
      const board = createBoard()
      setPiece(board, getSquareFromAlgebraic("f5"), Piece.DEFENDER)
      setPiece(board, getSquareFromAlgebraic("f4"), Piece.ATTACKER)

      const isPossible = isCapturePossible(
        board,
        getSquareFromAlgebraic("f5"),
        getSquareFromAlgebraic("f6"),
        getSquareFromAlgebraic("f4")
      )
      expect(isPossible).toBe(true)
    })

    it('defender can eat attacker near the throne', () => {
      const board = createBoard()
      setPiece(board, getSquareFromAlgebraic("f5"), Piece.ATTACKER)
      setPiece(board, getSquareFromAlgebraic("f4"), Piece.DEFENDER)

      const isPossible = isCapturePossible(
        board,
        getSquareFromAlgebraic("f5"),
        getSquareFromAlgebraic("f6"),
        getSquareFromAlgebraic("f4")
      )
      expect(isPossible).toBe(true)
    })
  })

  it('king cant eat defenders on corners (because one side)', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic("a2"), Piece.DEFENDER)
    setPiece(board, getSquareFromAlgebraic("a3"), Piece.KING)

    const isPossible = isCapturePossible(
      board,
      getSquareFromAlgebraic("a2"),
      getSquareFromAlgebraic("a1"),
      getSquareFromAlgebraic("a3")
    )
    expect(isPossible).toBe(false)
  })

  it('no capture when both sides are not same', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic("c2"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("c4"), Piece.DEFENDER)
    setPiece(board, getSquareFromAlgebraic("c3"), Piece.DEFENDER)

    const isPossible = isCapturePossible(
      board,
      getSquareFromAlgebraic("c3"),
      getSquareFromAlgebraic("c2"),
      getSquareFromAlgebraic("c4")
    )
    expect(isPossible).toBe(false)
  })

  it('no capture if all pieces is same', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic("c2"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("c4"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("c3"), Piece.ATTACKER)

    const isPossible = isCapturePossible(
      board,
      getSquareFromAlgebraic("c3"),
      getSquareFromAlgebraic("c2"),
      getSquareFromAlgebraic("c4")
    )
    expect(isPossible).toBe(false)
  })

  it('no capture when between is empty', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic("c2"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("c4"), Piece.ATTACKER)

    const isPossible = isCapturePossible(
      board,
      getSquareFromAlgebraic("c3"),
      getSquareFromAlgebraic("c2"),
      getSquareFromAlgebraic("c4")
    )
    expect(isPossible).toBe(false)
  })

  it('no capture when king on throne and try to capture defender', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic("f6"), Piece.KING)
    setPiece(board, getSquareFromAlgebraic("f5"), Piece.DEFENDER)
    setPiece(board, getSquareFromAlgebraic("f4"), Piece.ATTACKER)

    const isPossible = isCapturePossible(
      board,
      getSquareFromAlgebraic("f5"),
      getSquareFromAlgebraic("f6"),
      getSquareFromAlgebraic("f4")
    )
    expect(isPossible).toBe(false)
  })

  it('capture when king on throne and try to capture attacker', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic("f6"), Piece.KING)
    setPiece(board, getSquareFromAlgebraic("f5"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("f4"), Piece.DEFENDER)

    const isPossible = isCapturePossible(
      board,
      getSquareFromAlgebraic("f5"),
      getSquareFromAlgebraic("f6"),
      getSquareFromAlgebraic("f4")
    )
    expect(isPossible).toBe(false)
  })
})
