import {clearBoard, createBoard, setPiece} from "./board";
import {createMoveGenerator} from "./movegen";
import {Piece, Side} from "./types";
import {getSquareFromAlgebraic} from "./utils";
import {getMoveAlg} from "./print";
import {initLineMoves} from "./attackers";
import {beforeAll, describe, expect, it} from "vitest";

describe('Move Generation Tests', () => {
  const prepare = () => {
    const board = createBoard();
    clearBoard(board);

    const generator = createMoveGenerator();

    return {
      generator,
      board,
      printMoves: () => {
        for(let i = 0; i < generator.movesCount; ++i) {
          console.log(`${i + 1}. ${getMoveAlg(generator.moves[i])}`);
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
          try {
            expect(actualMoves.has(move)).toBe(true);
          } catch (e) {
            console.log(`Expected move not found: ${move}`);
            throw e;
          }
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
    const {board, generator, expectMovesCount} = prepare();
    generator.movegen(board);

    expectMovesCount(0);
  })

  it('1 piece on b1', () => {
    const {board, generator, expectMovesCount, expectIsMovesExists, expectIsMovesNotExists} = prepare();
    board.sideToMove = Side.ATTACKERS;
    setPiece(board, getSquareFromAlgebraic("b1"), Piece.ATTACKER);
    generator.movegen(board);

    expectMovesCount(18);

    // Valid moves
    expectIsMovesExists([
      "b1c1", "b1d1", "b1e1", "b1f1", "b1g1", "b1h1", "b1i1", "b1j1", // horizontal
      "b1b2", "b1b3", "b1b4", "b1b5", "b1b6", "b1b7", "b1b8", "b1b9", "b1b10", "b1b11"  // vertical
    ]);

    // Out of bounds
    expectIsMovesNotExists(["b1a1", "b1a10"]);
  })

  it('1 piece on b1 and enemy piece on c1', () => {
    const {board, generator, expectMovesCount, expectIsMovesExists, expectIsMovesNotExists} = prepare();
    board.sideToMove = Side.ATTACKERS;
    setPiece(board, getSquareFromAlgebraic("b1"), Piece.ATTACKER);
    setPiece(board, getSquareFromAlgebraic("c1"), Piece.DEFENDER);
    generator.movegen(board);

    expectMovesCount(10);

    // Valid moves
    expectIsMovesExists([
      "b1b2", "b1b3", "b1b4", "b1b5", "b1b6", "b1b7", "b1b8", "b1b9", "b1b10", "b1b11"  // vertical only
    ]);
  })
})
