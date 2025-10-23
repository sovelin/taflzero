import {clearBoard, createBoard, setPiece} from "./board";
import {createMoveGenerator} from "./movegen";
import {Piece, Side} from "./types";
import {getSquareFromAlgebraic} from "./utils";
import {getMoveAlg, printBoard} from "./print";
import {initLineMoves} from "./attackers";
import {beforeAll, describe, expect, it} from "vitest";

const printMoves = (moves: Uint32Array, count: number) => {
  for(let i = 0; i < count; ++i) {
    console.log(`${i}. ${getMoveAlg(moves[i])}`);
  }
}

describe('Move Generation Tests', () => {
  const createGenerator = () => {
    const generator = createMoveGenerator();

    return {
      generator,
      printMoves: () => {
        for(let i = 0; i < generator.movesCount; ++i) {
          console.log(`${i}. ${getMoveAlg(generator.moves[i])}`);
        }
      },
      expectMovesCount: (expectedCount: number) => {
        expect(generator.movesCount).toBe(expectedCount);
      },
      expectIsMovesExists: (expectedMoves: string[]) => {
        const actualMoves = new Set<string>();
        for(let i = 0; i < generator.movesCount; ++i) {
          actualMoves.add(getMoveAlg(generator.moves[i]));
        }
        for (const move of expectedMoves) {
          expect(actualMoves.has(move)).toBe(true);
        }
      },
      expectIsMovesNotExists: (notExpectedMoves: string[]) => {
        const actualMoves = new Set<string>();
        for(let i = 0; i < generator.movesCount; ++i) {
          actualMoves.add(getMoveAlg(generator.moves[i]));
        }
        for (const move of notExpectedMoves) {
          try {
            expect(actualMoves.has(move)).toBe(false);
          } catch (e) {
            console.log(`Unexpected move found: ${move}`);
            throw e;
          }
        }
      }
    }
  }

  beforeAll(() => {
    console.log('lol')
    initLineMoves()
  })

  it('clear board: no moves', () => {
    const generator = createMoveGenerator();
    const board = createBoard();
    generator.movegen(board);
    expect(generator.movesCount).toBe(0);
  })

  it('1 piece on the board', () => {
    const {generator, expectMovesCount, expectIsMovesExists, expectIsMovesNotExists} = createGenerator();
    const board = createBoard();

    board.sideToMove = Side.ATTACKERS;
    setPiece(board, getSquareFromAlgebraic("a1"), Piece.ATTACKER);
    generator.movegen(board);
    expectMovesCount(20);

    expectIsMovesExists(["a1a2", "a1a3", "a1a4", "a1a5", "a1a6", "a1a7", "a1a8", "a1a9", "a1a10"]);
    expectIsMovesExists(["a1b1", "a1c1", "a1d1", "a1e1", "a1f1", "a1g1", "a1h1", "a1i1", "a1j1"]);
    // Out of bounds
    expectIsMovesNotExists(["a1a1", "a1a10", "a1j1"]);
  })
})
