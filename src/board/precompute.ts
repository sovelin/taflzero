import {
  getAllNeighbors,
  getBottomLeftNeighbor,
  getBottomLeftSquare, getBottomNeighbor, getBottomRightNeighbor, getBottomRightSquare,
  getCol,
  getCornersSq, getLeftNeighbor, getRightNeighbor,
  getRow,
  getThroneSq, getTopLeftNeighbor,
  getTopLeftSquare, getTopNeighbor, getTopRightNeighbor,
  getTopRightSquare, getVerticalHorizontalNeighbors
} from "./utils";
import {SQS} from "@/board/constants";

export const ROW = new Uint8Array(SQS);
export const COL = new Uint8Array(SQS);
export const THRONE_SQ = getThroneSq();
export const CORNERS_SQ = getCornersSq();
export const TOP_LEFT_SQ = getTopLeftSquare()
export const TOP_RIGHT_SQ = getTopRightSquare()
export const BOTTOM_LEFT_SQ = getBottomLeftSquare()
export const BOTTOM_RIGHT_SQ = getBottomRightSquare()

export const RIGHT_NEIGHBOR: (number | null)[] = Array.from({length: SQS})
export const LEFT_NEIGHBOR: (number | null)[] = Array.from({length: SQS})
export const TOP_NEIGHBOR: (number | null)[] = Array.from({length: SQS})
export const BOTTOM_NEIGHBOR: (number | null)[] = Array.from({length: SQS})
export const TOP_RIGHT_NEIGHBOR: (number | null)[] = Array.from({length: SQS})
export const TOP_LEFT_NEIGHBOR: (number | null)[] = Array.from({length: SQS})
export const BOTTOM_RIGHT_NEIGHBOR: (number | null)[] = Array.from({length: SQS})
export const BOTTOM_LEFT_NEIGHBOR: (number | null)[] = Array.from({length: SQS})
export const VERTICAL_HORIZONTAL_NEIGHBORS: number[][] = Array.from({length: SQS})
export const ALL_NEIGHBORS: number[][] = Array.from({length: SQS})

export const precomputeBoard = () => {
  for (let i = 0; i < SQS; i++) {
    ROW[i] = getRow(i);
    COL[i] = getCol(i);
    RIGHT_NEIGHBOR[i] = getRightNeighbor(i)
    LEFT_NEIGHBOR[i] = getLeftNeighbor(i)
    TOP_NEIGHBOR[i] = getTopNeighbor(i)
    BOTTOM_NEIGHBOR[i] = getBottomNeighbor(i)
    TOP_RIGHT_NEIGHBOR[i] = getTopRightNeighbor(i)
    TOP_LEFT_NEIGHBOR[i] = getTopLeftNeighbor(i)
    BOTTOM_RIGHT_NEIGHBOR[i] = getBottomRightNeighbor(i)
    BOTTOM_LEFT_NEIGHBOR[i] = getBottomLeftNeighbor(i)
    VERTICAL_HORIZONTAL_NEIGHBORS[i] = getVerticalHorizontalNeighbors(i)
    ALL_NEIGHBORS[i] = getAllNeighbors(i)
  }
}
