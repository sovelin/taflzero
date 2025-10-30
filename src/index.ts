import {createMoveGenerator, initMovesModule, makeMove} from "@/moves";
import {Board, createBoard, getSquareFromAlgebraic, Piece, setInitialPosition, setPiece, Side} from "@/board";
import {printBoard} from "@/board/print";
import {search} from "@/search/search";
import {statistics} from "@/search";
import {checkTerminal, getScoreText} from "@/evaluation";
import {bestMove} from "@/search/model/BestMove";
import {getMoveAlg} from "@/moves/utils";
import {createMoveFromAlgebraic} from "@/moves/move";

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

const setLostPosition = (board: Board) => {
  setPiece(board, getSquareFromAlgebraic("b11"), Piece.KING);
  setPiece(board, getSquareFromAlgebraic("d5"), Piece.ATTACKER);
  board.sideToMove = 1;
}

const setNotFullyLostPosition = (board: Board) => {
  setPiece(board, getSquareFromAlgebraic("c11"), Piece.KING);
  setPiece(board, getSquareFromAlgebraic("b11"), Piece.ATTACKER);
  setPiece(board, getSquareFromAlgebraic("e11"), Piece.ATTACKER);
  board.sideToMove = Side.DEFENDERS;
}

const setMoreDifficultPosition = (board: Board) => { // best should be c8c11
  setPiece(board, getSquareFromAlgebraic("c8"), Piece.KING);
  setPiece(board, getSquareFromAlgebraic("b11"), Piece.ATTACKER);
  setPiece(board, getSquareFromAlgebraic("a8"), Piece.ATTACKER);
  setPiece(board, getSquareFromAlgebraic("e8"), Piece.ATTACKER);
  setPiece(board, getSquareFromAlgebraic("c7"), Piece.ATTACKER);
  setPiece(board, getSquareFromAlgebraic("e11"), Piece.ATTACKER);
  board.sideToMove = Side.DEFENDERS;
}

const runAlphaBetaTest = () => {
  const board = createBoard()
  //setInitialPosition(board);
  //setLostPosition(board)
  //setNotFullyLostPosition(board)
  setMoreDifficultPosition(board)

  const depth = 7;
  const res = search(board, depth);

  console.log(`Search result at depth ${depth}: ${getScoreText(res)}`);
  console.log(`Searched nodes: ${statistics.nodes}`)
  console.log(`Best move: ${getMoveAlg(bestMove.move)}`)
  printBoard(board)
}

//runSimulator();
runAlphaBetaTest();

/*
*
*
* Search result at depth 7: mate in 5
Searched nodes: 2671850
Best move: c8c9 // incorrect! could be faster with c8c11
    a  b  c  d  e  f  g  h  i  j  k
  ┌────────────────────────────────┐
11│ .  A  .  .  A  .  .  .  .  .  .│11
10│ .  .  .  .  .  .  .  .  .  .  .│10
 9│ .  .  .  .  .  .  .  .  .  .  .│ 9
 8│ .  A  K  .  A  .  .  .  .  .  .│ 8
 7│ .  .  A  .  .  .  .  .  .  .  .│ 7
 6│ .  .  .  .  .  .  .  .  .  .  .│ 6
 5│ .  .  .  .  .  .  .  .  .  .  .│ 5
 4│ .  .  .  .  .  .  .  .  .  .  .│ 4
 3│ .  .  .  .  .  .  .  .  .  .  .│ 3
 2│ .  .  .  .  .  .  .  .  .  .  .│ 2
 1│ .  .  .  .  .  .  .  .  .  .  .│ 1
  └────────────────────────────────┘
    a  b  c  d  e  f  g  h  i  j  k
*
* */
