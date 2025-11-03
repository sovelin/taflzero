import {Board, Side} from "@/board";
import {createMoveGenerator, getMoveAlg, makeMove, unmakeMove} from "@/moves";
import {statistics} from "@/search/model/Statistics";
import {MATE_SCORE, checkTerminal, evaluateBoard, sidedEval} from "@/evaluation";
import {bestMove} from "@/search/model/BestMove";
import {createTranspositionTable} from "@/transposition/createTT";
import {TTFlag} from "@/transposition";
import {timer} from "@/search/model/Timer";
import {MoveGenerator} from "@/moves/movegen/movegen";

const MAX_DEPTH = 256;
const moveGens = Array.from({length: MAX_DEPTH}, () => createMoveGenerator());

const tt = createTranspositionTable()

const moveGenAtDepth = (depth: number) => {
  return moveGens[depth];
}

const isTTMoveValid = (moveGen: MoveGenerator, ttMove: number | null) => {
  for(let i = 0; i < moveGen.movesCount; i++) {
    if (moveGen.moves[i] === ttMove) {
      return true;
    }
  }

  return false;
}

export const search = (
  board: Board,
  depth: number,
  alpha: number = -MATE_SCORE,
  beta: number = MATE_SCORE
  , height = 0
) => {
  const terminal = checkTerminal(board);

  if (terminal !== null) {
    return sidedEval(board, (terminal as Side) === Side.DEFENDERS ? MATE_SCORE - height : -MATE_SCORE + height);
  }

  if (depth === 0) {
    return evaluateBoard(board);
  }

  const zobrist = board.zobrist;
  const ttEntry = tt.probe(zobrist);
  if (ttEntry && ttEntry.depth >= depth && height > 0) {
    if (ttEntry.flag === TTFlag.EXACT) {
      return ttEntry.score;
    }

    if (ttEntry.flag === TTFlag.LOWERBOUND) {
      alpha = Math.max(alpha, ttEntry.score);
    } else if (ttEntry.flag === TTFlag.UPPERBOUND) {
      beta = Math.min(beta, ttEntry.score);
    }

    if (alpha >= beta) {
      return ttEntry.score;
    }
  }

  if (timer.isTimeUp()) {
    return 0;
  }

  const moveGen = moveGenAtDepth(height);
  moveGen.movegen(board);

  let ttType = TTFlag.UPPERBOUND;
  let ttMove: number | null = null;

  for (let i = -1; i < moveGen.movesCount; i++) {
    const move: number =  i === -1 ? (ttEntry?.move || 0) : moveGen.moves[i];

    if (i === -1) {
      if (!isTTMoveValid(moveGen, ttEntry?.move || null)) {
        continue;
      }
    }

    statistics.incrementNodes();

    // Make the move
    const undo = makeMove(board, move);

    if (height === 0) {
      console.log(`Move number: ${i}; Move: ${getMoveAlg(move)}`);
    }

    // Recursively search
    const score = -search(
      board,
      depth - 1,
      -beta,
      -alpha,
      height + 1
    );

    // Undo the move
    unmakeMove(board, undo);

    if (timer.isTimeUp()) {
      break
    }

    if (score > alpha) {
      alpha = score;
      if (height === 0) {
        bestMove.setBestMove(move);
      }
      ttType = TTFlag.EXACT;
      ttMove = move;
    }

    if (alpha >= beta) {
      ttType = TTFlag.LOWERBOUND;
      break; // Beta cutoff
    }
  }

  tt.store(zobrist, depth, alpha, ttType, ttMove || 0);

  return alpha;
}
