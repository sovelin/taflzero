import {createMoveGenerator, initMovesModule, makeMove} from "@/moves";
import {createBoard, setInitialPosition} from "@/board";
import {printBoard} from "@/board/print";

initMovesModule()

const runSimulator = () => {
  const board = createBoard()
  setInitialPosition(board);
  const generator = createMoveGenerator()

  const random = (from: number, to: number) => {
    return Math.floor(Math.random() * (to - from)) + from;
  }

  const makeNextMove = () => {
    generator.movegen(board);
    console.log(`moves count: ${generator.movesCount}`);
    console.log({board})
    const nextMove = generator.moves[random(0, generator.movesCount)];
    makeMove(board, nextMove);
  }

  const interval = setInterval(() => {
    makeNextMove();
    console.log('--- New Board ---')
    printBoard(board);

    if (generator.movesCount === 0) {
      clearInterval(interval);
    }
  }, 100);
}
runSimulator();
