import {MATE_SCORE} from "./constants";

const MATE_SCORE_THRESHOLD = MATE_SCORE - 256;

export const isMateScore = (score: number): boolean => {
  return Math.abs(score) >= MATE_SCORE_THRESHOLD;
}
