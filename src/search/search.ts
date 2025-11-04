import {Board, Side} from "@/board";
import {createMoveGenerator, makeMove, unmakeMove} from "@/moves";
import {statistics} from "@/search/model/Statistics";
import {MATE_SCORE, checkTerminal, evaluateBoard, sidedEval, isMateScore} from "@/evaluation";
import {bestMove} from "@/search/model/BestMove";
import {createTranspositionTable} from "@/transposition/createTT";
import {TTFlag} from "@/transposition";
import {timer} from "@/search/model/Timer";
import {MoveGenerator} from "@/moves/movegen/movegen";
import {estimateMoves, MoveScores, pickMove} from "@/search/movesOrdering";
import {readScore} from "@/transposition/utils";

const MAX_DEPTH = 256;
const moveGens = Array.from({length: MAX_DEPTH}, () => createMoveGenerator());

export const tt = createTranspositionTable()

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
  if (ttEntry && ttEntry.zobrist === zobrist && ttEntry.depth >= depth && height > 0) {
    let readedScore = readScore(ttEntry.score, height);

    if (ttEntry.flag === TTFlag.EXACT) {
      return readedScore;
    }

    if (ttEntry.flag === TTFlag.LOWERBOUND) {
      alpha = Math.max(alpha, readedScore);
    } else if (ttEntry.flag === TTFlag.UPPERBOUND) {
      beta = Math.min(beta, readedScore);
    }

    if (alpha >= beta) {
      return readedScore;
    }
  }

  if (timer.isTimeUp()) {
    return 0;
  }

  const moveGen = moveGenAtDepth(height);
  moveGen.movegen(board);

  estimateMoves(moveGen, MoveScores[height], moveGen.movesCount, ttEntry?.move || null);

  let move;

  let ttType = TTFlag.UPPERBOUND;
  let ttMove: number | null = null;

  while (move = pickMove(moveGen, MoveScores[height])) {
    statistics.incrementNodes();

    // Make the move
    const undo = makeMove(board, move);

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

  tt.store(zobrist, depth, alpha, ttType, ttMove || 0, height);

  return alpha;
}
