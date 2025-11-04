import {isMateScore} from "@/evaluation";

export const readScore = (score: number, height: number) => {
  if (isMateScore(score)) {
    if (score > 0) {
      return score - height;
    } else {
      return score + height;
    }
  }

  return score
}

export const writeScore = (score: number, height: number) => {
  if (isMateScore(score)) {
    if (score > 0) {
      return score + height;
    } else {
      return score - height;
    }
  }

  return score
}
