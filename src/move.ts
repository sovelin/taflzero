import {Board} from "./model/Board";

export function createMove(from: number, to: number) {
  return from << 16 | to;
}

export function moveFrom(move: number) {
  return (move >> 16) & 0xFFFF;
}

export function moveTo(move: number) {
  return move & 0xFFFF;
}
