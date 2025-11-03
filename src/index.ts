import {createMoveGenerator, initMovesModule, makeMove} from "@/moves";
import {Board, createBoard, getSquareFromAlgebraic, Piece, setInitialPosition, setPiece, Side} from "@/board";
import {printBoard} from "@/board/print";
import {search} from "@/search/search";
import {searchRoot, statistics} from "@/search";
import {checkTerminal, getScoreText} from "@/evaluation";
import {bestMove} from "@/search/model/BestMove";
import {getMoveAlg} from "@/moves";

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

const setWinningPositionForAttackers = (board: Board) => {
  setPiece(board, getSquareFromAlgebraic("b10"), Piece.KING);
  setPiece(board, getSquareFromAlgebraic("a10"), Piece.ATTACKER);
  setPiece(board, getSquareFromAlgebraic("c10"), Piece.ATTACKER);
  setPiece(board, getSquareFromAlgebraic("h11"), Piece.ATTACKER);
  setPiece(board, getSquareFromAlgebraic("b9"), Piece.ATTACKER);
  board.sideToMove = Side.ATTACKERS;
}

const setCloseToSurroundedPosition = (board: Board) => {
  setPiece(board, getSquareFromAlgebraic('a10'), Piece.ATTACKER)
  setPiece(board, getSquareFromAlgebraic('b11'), Piece.ATTACKER)
  setPiece(board, getSquareFromAlgebraic('c11'), Piece.ATTACKER)
  setPiece(board, getSquareFromAlgebraic('d8'), Piece.ATTACKER)
  setPiece(board, getSquareFromAlgebraic('c9'), Piece.ATTACKER)
  setPiece(board, getSquareFromAlgebraic('b9'), Piece.ATTACKER)

  setPiece(board, getSquareFromAlgebraic('b10'), Piece.KING)

  board.sideToMove = Side.ATTACKERS;
}

const sertNearToFortPosition = (board: Board) => {
  board.sideToMove = Side.ATTACKERS;
  setPiece(board, getSquareFromAlgebraic('f2'), Piece.KING)
  setPiece(board, getSquareFromAlgebraic('b1'), Piece.DEFENDER)
  setPiece(board, getSquareFromAlgebraic('b6'), Piece.ATTACKER)
  setPiece(board, getSquareFromAlgebraic('a5'), Piece.ATTACKER)
  setPiece(board, getSquareFromAlgebraic('e1'), Piece.DEFENDER)
  setPiece(board, getSquareFromAlgebraic('e2'), Piece.DEFENDER)
  setPiece(board, getSquareFromAlgebraic('e3'), Piece.DEFENDER)
  setPiece(board, getSquareFromAlgebraic('f4'), Piece.DEFENDER)
  setPiece(board, getSquareFromAlgebraic('g3'), Piece.DEFENDER)
  setPiece(board, getSquareFromAlgebraic('d2'), Piece.DEFENDER)
  setPiece(board, getSquareFromAlgebraic('h2'), Piece.DEFENDER)
  setPiece(board, getSquareFromAlgebraic('i2'), Piece.DEFENDER)
  setPiece(board, getSquareFromAlgebraic('c2'), Piece.DEFENDER)
  board.lastMoveTo = getSquareFromAlgebraic('b1');
  printBoard(board)
}

const runAlphaBetaTest = () => {
  const board = createBoard()
  setInitialPosition(board);
  printBoard(board)
  //setLostPosition(board)
  //setNotFullyLostPosition(board)
  //setMoreDifficultPosition(board)

  const depth = 5;
  const res = search(board, depth);

  console.log(`Search result at depth ${depth}: ${getScoreText(res)}`);
  console.log(`Searched nodes: ${statistics.nodes}`)
  console.log(`Best move: ${getMoveAlg(bestMove.move)}`)
  printBoard(board)
}

const runSelfPlayTest = () => {
  const board = createBoard()
  setInitialPosition(board);
  //setWinningPositionForAttackers(board)
  //setCloseToSurroundedPosition(board)
  //sertNearToFortPosition(board)
  printBoard(board);

  const maxDepth = 5;

  let movesCount = 0
  while (true) {
    movesCount++;
    const {bestMove, bestScore} = searchRoot(board, {
      time: 1000,
      onIteration: (depth, move, score, nodes: number, speed: number, time: number) => {
        console.log(`Depth: ${depth}, Move: ${getMoveAlg(move)}, Score: ${getScoreText(score)}, Nodes: ${nodes}, Speed (knps): ${Math.floor(speed / 1000)}, Time (ms): ${time}`);
      }
    });

    makeMove(board, bestMove);

    console.log(`Best move: ${getMoveAlg(bestMove)}`);
    printBoard(board);

    if (checkTerminal(board) !== null) {
      console.log(board.repTable)
      console.log('--- Game Over ---');
      break;
    }
  }
}

runSelfPlayTest();
//runAlphaBetaTest()
setInterval(() => {}, 1e9);

