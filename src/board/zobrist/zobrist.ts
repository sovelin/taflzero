import {NUM_PIECE_KINDS, SQS} from "../constants";
import {random64} from "@/utils/random";

export const zobrist: bigint[][] = Array.from({ length: NUM_PIECE_KINDS }, () =>
  Array.from({ length: SQS }, () => random64())
);

export const sideZobrist = random64();

