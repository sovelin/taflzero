import {createBoard, setInitialPosition} from "../../engine/src/board";
import {printBoard} from "./print";
import {initLineMoves, printLineMaskExamples} from "../../engine/src/attackers";


const board = createBoard();
setInitialPosition(board);
printBoard(board);

initLineMoves();

printLineMaskExamples(5, 0b00000111110);
printLineMaskExamples(3, 0b00011001100);
