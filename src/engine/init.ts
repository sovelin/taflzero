import {initMovesModule} from "@/moves";
import {precomputeBoard} from "@/board";

export const initEngine = () => {
  initMovesModule()
  precomputeBoard()
}
