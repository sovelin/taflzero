import {createMoveGenerator} from "@/moves";
import {pickMove} from "@/search/movesOrdering";
import {describe, it, expect} from 'vitest'

describe('moveOrdering', () => {
  it('should pick moves in corrrect order', () => {
    const moveGen = createMoveGenerator()

    moveGen.moves[0] = 1
    moveGen.moves[1] = 2
    moveGen.moves[2] = 3
    moveGen.moves[3] = 4
    moveGen.moves[4] = 5

    moveGen.setMovesCount(5)

    const moveScores = new Int32Array(5)
    moveScores[0] = 10
    moveScores[1] = 50
    moveScores[2] = 20
    moveScores[3] = 40
    moveScores[4] = 30
    moveScores[5] = 777

    const pickedMoves = []
    while (moveGen.movesCount > 0) {
      const move = pickMove(moveGen, moveScores)
      pickedMoves.push(move)
    }

    expect(pickedMoves).toEqual([2, 4, 5, 3, 1])
  })
})
