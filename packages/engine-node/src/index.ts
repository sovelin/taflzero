import {createBoard, setInitialPosition} from "../../engine/src/board";
import {getMoveAlg, printBoard} from "./print";
import {initLineMoves, printLineMaskExamples} from "../../engine/src/attackers";
import {getMovesCount, movegen, moves} from "../../engine/src/movegen";


const board = createBoard();
setInitialPosition(board);
printBoard(board);

initLineMoves();

printLineMaskExamples(5, 0b00000111110);
printLineMaskExamples(3, 0b00011001100);

movegen(board)
console.log(`Generated ${getMovesCount()} moves`)

for (let i = 0; i < getMovesCount(); i++) {
  console.log(`Move ${i + 1}: ${getMoveAlg(moves[i])}`);
}
