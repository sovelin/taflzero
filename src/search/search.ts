import {Board, Side} from "@/board";
import {createMoveGenerator, makeMove, unmakeMove} from "@/moves";
import {statistics} from "@/search/model/Statistics";
import {MATE_SCORE, checkTerminal, isMateScore, evaluateBoard, sidedEval} from "@/evaluation";
import {bestMove} from "@/search/model/BestMove";
import {getMoveAlg} from "@/moves/utils";



const MAX_DEPTH = 256;
const moveGens = Array.from({length: MAX_DEPTH}, () => createMoveGenerator());

const moveGenAtDepth = (depth: number) => {
  return moveGens[depth];
}

export const search = (
  board: Board,
  depth: number,
  alpha: number = -Infinity,
  beta: number = Infinity
  , height = 0
) => {
  if (height === 0) {
    statistics.reset()
  }

  const terminal = checkTerminal(board);

  if (terminal) {
    return sidedEval(board, (terminal as Side) === Side.DEFENDERS ? MATE_SCORE - height : -MATE_SCORE + height);
  }

  if (depth === 0) {
    return evaluateBoard(board);
  }

  const moveGen = moveGenAtDepth(depth);
  moveGen.movegen(board);

  for (let i = 0; i < moveGen.movesCount; i++) {
    const move = moveGen.moves[i];
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

    if (score > alpha) {
      alpha = score;
      if (height === 0) {
        bestMove.setBestMove(move);
      }
    }

    if (alpha >= beta) {
      break; // Beta cutoff
    }
  }

  return alpha;
}
