import {moveFrom, moveTo} from "@/moves/move";
import {getSquareAlg} from "@/board/utils";

export function getMoveAlg(move: number) {
  return `${getSquareAlg(moveFrom(move))}${getSquareAlg(moveTo(move))}`;
}
