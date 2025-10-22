import {createBoard, setInitialPosition} from "../../engine/src/board";
import {getMoveAlg, printBoard} from "./print";
import {initLineMoves, printLineMaskExamples} from "../../engine/src/attackers";
import {createMoveGenerator} from "../../engine/src/movegen";


const board = createBoard();
setInitialPosition(board);
printBoard(board);

initLineMoves();

printLineMaskExamples(5, 0b00000111110);
printLineMaskExamples(3, 0b00011001100);

const moveGenerator = createMoveGenerator()

moveGenerator.movegen(board);
console.log(`Generated ${moveGenerator.movesCount} moves`)

for (let i = 0; i < moveGenerator.movesCount; i++) {
  console.log(`Move ${i + 1}: ${getMoveAlg(moveGenerator.moves[i])}`);
}
