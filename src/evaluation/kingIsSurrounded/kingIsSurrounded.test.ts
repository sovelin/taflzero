import {describe, it, expect} from 'vitest'
import {createBoard, getSquareFromAlgebraic, Piece, setPiece} from "@/board";
import {kingIsSurrounded} from "./kingIsSurrounded";

describe('King Is Surrounded Evaluation Tests', () => {
  it('king is not surrounded', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic("b2"), Piece.KING)

    const isSurrounded = kingIsSurrounded(board);
    expect(isSurrounded).toBeFalsy()
  })

  it('surrounded by attackers not from all sides -> not surrounded', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic("b2"), Piece.KING)
    setPiece(board, getSquareFromAlgebraic("a2"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("b1"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("c2"), Piece.ATTACKER)

    const isSurrounded = kingIsSurrounded(board);
    expect(isSurrounded).toBeFalsy()
  })

  it('surrounded by attackers from all sides -> surrounded', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic("b2"), Piece.KING)
    setPiece(board, getSquareFromAlgebraic("a2"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("b1"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("c2"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("b3"), Piece.ATTACKER)

    const isSurrounded = kingIsSurrounded(board);
    expect(isSurrounded).toBeTruthy()
  })

  it('surrounded by defenders from all sides -> not surrounded', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic("b2"), Piece.KING)
    setPiece(board, getSquareFromAlgebraic("a2"), Piece.DEFENDER)
    setPiece(board, getSquareFromAlgebraic("b1"), Piece.DEFENDER)
    setPiece(board, getSquareFromAlgebraic("c2"), Piece.DEFENDER)
    setPiece(board, getSquareFromAlgebraic("b3"), Piece.DEFENDER)

    const isSurrounded = kingIsSurrounded(board);
    expect(isSurrounded).toBeFalsy()
  })

  it('surrounded by 3 attackers on board side -> not surrounded', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic("e1"), Piece.KING)
    setPiece(board, getSquareFromAlgebraic("d1"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("f1"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("e2"), Piece.ATTACKER)

    const isSurrounded = kingIsSurrounded(board);
    expect(isSurrounded).toBeFalsy()
  })

  it('surrounded by 3 attackers and by throne -> surrounded', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic("e6"), Piece.KING)
    setPiece(board, getSquareFromAlgebraic("e7"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("e5"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("d6"), Piece.ATTACKER)

    const isSurrounded = kingIsSurrounded(board);
    expect(isSurrounded).toBeTruthy()
  })

  it('surrounded by corner and by corner -> not surrounded', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic("b1"), Piece.KING)
    setPiece(board, getSquareFromAlgebraic("b2"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("c1"), Piece.ATTACKER)

    const isSurrounded = kingIsSurrounded(board);
    expect(isSurrounded).toBeFalsy()
  })

  it('on throne and surrounded by 4 attackers -> surrounded', () => {
    const board = createBoard()
    setPiece(board, getSquareFromAlgebraic("f6"), Piece.KING)
    setPiece(board, getSquareFromAlgebraic("f5"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("f7"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("e6"), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic("g6"), Piece.ATTACKER)

    const isSurrounded = kingIsSurrounded(board);
    expect(isSurrounded).toBeTruthy()
  })
})
