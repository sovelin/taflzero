import {createBoard, setInitialPosition} from "../../engine/src/board";
import {printBoard} from "./print";


const board = createBoard();
setInitialPosition(board);
printBoard(board);
