import {createMoveGenerator} from "./movegen";
import {initLineMoves, printLineMaskExamples} from "./attackers";
import {createBoard, setInitialPosition} from "./board";
import {getMoveAlg} from "./print";

const board = createBoard();
setInitialPosition(board);

function printBoard(board: any) {

}

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
