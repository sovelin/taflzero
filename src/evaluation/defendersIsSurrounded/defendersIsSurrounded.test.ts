import {describe, it, expect} from "vitest";
import {createBoard, getSquareFromAlgebraic, Piece, setPiece} from "@/board";
import {defendersIsSurrounded} from "@/evaluation/defendersIsSurrounded/defendersIsSurrounded";
import {printBoard} from "@/board/print";

describe('Defenders Is Surrounded Evaluation Tests', () => {
    it('by default not surrounded', () => {
      const board = createBoard()
      setPiece(board, getSquareFromAlgebraic('d4'), Piece.KING)

      const isSurrounded = defendersIsSurrounded(board)

      expect(isSurrounded).toBeFalsy()
    })

  it('surrounded corners but with open edges -> not surrounded', () => {
    const board = createBoard()

    setPiece(board, getSquareFromAlgebraic('a2'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('b1'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('j1'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('k2'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('a10'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('b11'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('j11'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('k10'), Piece.ATTACKER)

    setPiece(board, getSquareFromAlgebraic('d4'), Piece.KING)

    const isSurrounded = defendersIsSurrounded(board)

    expect(isSurrounded).toBeFalsy()
  })

  it('surrounded corners and edges plus no one piece available out of surrounding -> surrounded', () => {
    const board = createBoard()

    setPiece(board, getSquareFromAlgebraic('a10'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('b11'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('c11'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('d10'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('c9'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('b9'), Piece.ATTACKER)

    setPiece(board, getSquareFromAlgebraic('b10'), Piece.KING)

    const isSurrounded = defendersIsSurrounded(board)

    expect(isSurrounded).toBeTruthy()
  })

  it('surrounded corners and edges, one piece available out of surrounding -> not surrounded', () => {
    const board = createBoard()

    setPiece(board, getSquareFromAlgebraic('a10'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('b11'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('c11'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('d10'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('c9'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('b9'), Piece.ATTACKER)

    setPiece(board, getSquareFromAlgebraic('b10'), Piece.KING)
    setPiece(board, getSquareFromAlgebraic("j1"), Piece.DEFENDER)
    printBoard(board)

    const isSurrounded = defendersIsSurrounded(board)

    expect(isSurrounded).toBeFalsy()
  })

  it('surrounded corners and edges plus (but defender also in chain) -> not surrounded', () => {
    const board = createBoard()

    setPiece(board, getSquareFromAlgebraic('a10'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('b11'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('c11'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('d10'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('c9'), Piece.ATTACKER)
    setPiece(board, getSquareFromAlgebraic('b9'), Piece.DEFENDER)

    setPiece(board, getSquareFromAlgebraic('b10'), Piece.KING)

    const isSurrounded = defendersIsSurrounded(board)

    expect(isSurrounded).toBeFalsy()
  })
})
