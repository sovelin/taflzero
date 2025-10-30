import {isMateScore} from "@/evaluation/utils";
import {MATE_SCORE} from "@/evaluation/constants";

export const getScoreText = (score: number) => {
  if (isMateScore(score)) {
    return `mate in ${MATE_SCORE - Math.abs(score)}`
  }

  return String(score);
}
