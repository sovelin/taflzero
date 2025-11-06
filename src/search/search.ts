import {Board, Side} from "@/board";
import {createMoveGenerator, makeMove, moveFrom, moveTo, unmakeMove} from "@/moves";
import {statistics} from "@/search/model/Statistics";
import {MATE_SCORE, checkTerminal, evaluateBoard, sidedEval} from "@/evaluation";
import {bestMove} from "@/search/model/BestMove";
import {createTranspositionTable} from "@/transposition/createTT";
import {TTFlag} from "@/transposition";
import {timer} from "@/search/model/Timer";
import {estimateMoves, MoveScores, pickMove} from "@/search/movesOrdering";
import {readScore} from "@/transposition/utils";
import {MAX_DEPTH} from "./constants";
import {killers, saveKiller} from "./model/Killers";
import {updateHistory} from "@/search/model/History";


const moveGens = Array.from({length: MAX_DEPTH}, () => createMoveGenerator());

export const tt = createTranspositionTable()

const moveGenAtDepth = (depth: number) => {
  return moveGens[depth];
}

export const search = (
  board: Board,
  depth: number,
  alpha: number = -MATE_SCORE * 2,
  beta: number = MATE_SCORE * 2
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
  const isValidTT = ttEntry?.zobrist === zobrist

  const isPvNode = alpha !== beta - 1;

  if (!isPvNode && height > 0 && ttEntry && isValidTT && ttEntry.depth >= depth && height > 0) {
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
      return alpha;
    }
  }

  if (timer.isTimeUp()) {
    return 0;
  }

  const moveGen = moveGenAtDepth(height);
  moveGen.movegen(board);
  estimateMoves(moveGen, MoveScores[height], moveGen.movesCount, isValidTT ? (ttEntry?.move || null) : null, height, board.sideToMove);


  let ttType = TTFlag.UPPERBOUND;
  let ttMove: number | null = null;

  let i = 0;

  while (moveGen.movesCount > 0) {
    i += 1
    const move = pickMove(moveGen, MoveScores[height])
    statistics.incrementNodes();

    // Make the move
    const undo = makeMove(board, move);

    // Recursively search
    let score;
    if (i === 1) {
      score = -search(
        board,
        depth - 1,
        -beta,
        -alpha,
        height + 1
      );
    } else {
      score = -search(
        board,
        depth - 1,
        -alpha - 1,
        -alpha,
        height + 1
      );

      if (score > alpha && score < beta) {
        score = -search(
          board,
          depth - 1,
          -beta,
          -alpha,
          height + 1
        );
      }
    }

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
      updateHistory(board.sideToMove, moveFrom(move), moveTo(move), depth);
    }

    if (alpha >= beta) {
      ttType = TTFlag.LOWERBOUND;
      saveKiller(height, move);
      break; // Beta cutoff
    }
  }

  tt.store(zobrist, depth, alpha, ttType, ttMove || 0, height);

  return alpha;
}
