import {MAX_DEPTH} from "../constants";

export const killers = Array.from({length: MAX_DEPTH})
  .fill(null)
  .map(() => {
    return [0, 0];
  });

export const saveKiller = (height: number, move: number) => {
  const killer = killers[height];

  if (!killer[0]) {
    killer[0] = move;
  } else {
    killer[1] = killer[0];
    killer[0] = move;
  }
}

export const clearKillers = () => {
  for (let i = 0; i < MAX_DEPTH; i++) {
    killers[i][0] = 0;
    killers[i][1] = 0;
  }
}
