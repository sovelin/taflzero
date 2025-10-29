import {createMoveFromAlgebraic, moveFrom, moveTo} from "./move";
import {describe, it, expect} from "vitest";
import {getCol, getRow} from "@/board";

describe('Move tests', () => {
  it('algebraic move', () => {
    const move = createMoveFromAlgebraic('a10a9');
    const fromSq = moveFrom(move)
    const toSq = moveTo(move)

    const fromRow = getRow(fromSq)
    const toRow = getRow(toSq)

    const fromCol = getCol(fromSq)
    const toCol = getCol(toSq)

    expect(fromCol).toBe(0);
    expect(fromRow).toBe(9);
    expect(toCol).toBe(0);
    expect(toRow).toBe(8);
  })
})
