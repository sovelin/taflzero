import {initMovesModule} from "@/moves";
import {precomputeBoard} from "@/board";

export const initEngine = () => {
  console.log('here')
  initMovesModule()
  precomputeBoard()
}
