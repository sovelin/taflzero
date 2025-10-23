import {clearBoard, createBoard, setPiece} from "./board";
import {createMoveGenerator} from "./movegen";
import {Piece, Side} from "./types";
import {getSquareFromAlgebraic} from "./utils";
import {printBoard} from "./print";
import {initLineMoves} from "./attackers";
import {beforeAll, describe, expect, it} from "vitest";

describe('Move Generation Tests', () => {
  beforeAll(() => {
    console.log('lol')
    initLineMoves()
  })

  it('clear board: no moves', () => {
    const generator = createMoveGenerator();
    const board = createBoard();
    clearBoard(board);
    generator.movegen(board);
    expect(generator.movesCount).toBe(0);
  })

  it('2 pieces on the board', () => {
    const generator = createMoveGenerator();
    const board = createBoard();



    clearBoard(board);
    board.sideToMove = Side.ATTACKERS;
    setPiece(board, getSquareFromAlgebraic("a1"), Piece.ATTACKER);
    generator.movegen(board);
    printBoard(board)
    console.log(`Generated ${generator.movesCount} moves`);
    //expect(generator.movesCount).toBe(7);
  })
})
