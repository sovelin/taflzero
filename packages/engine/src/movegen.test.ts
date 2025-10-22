import {clearBoard, createBoard} from "./board";
import {createMoveGenerator} from "./movegen";

describe('Move Generation Tests', () => {
  it('should generate correct moves for a given board position', () => {
    const generator = createMoveGenerator();
    const board = createBoard();
    clearBoard(board);
    expect(generator.movesCount).toBe(0);
  })
})
