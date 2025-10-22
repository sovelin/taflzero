import {Board} from "../../engine/src/model/Board";
import {Piece} from "../../engine/src/types";
import {moveFrom, moveTo} from "../../engine/src/move";
import {getCol, getRow} from "../../engine/src/utils";

export function printBoard (b: Board) {
  const size = 11;
  const cols = "abcdefghijk".split("");
  const color = (txt: string, code: number) => `\x1b[${code}m${txt}\x1b[0m`;

  const sym = (p: number): string => {
    switch (p) {
      case Piece.ATTACKER: return color("A", 31); // красный
      case Piece.DEFENDER: return color("D", 34); // синий
      case Piece.KING: return color("K", 33);     // жёлтый
      default: return ".";
    }
  };

  const top = "    " + cols.map(c => `${c.padEnd(2)}`).join(" ") + "\n";
  const borderTop = "  ┌" + "─".repeat(size * 3 - 1) + "┐";
  const borderBottom = "  └" + "─".repeat(size * 3 - 1) + "┘";

  console.log(top + borderTop);

  for (let r = size - 1; r >= 0; r--) {
    let row = `${(r + 1).toString().padStart(2)}│ `;
    for (let c = 0; c < size; c++) {
      const sq = r * size + c;
      row += sym(b.board[sq]);

      if (c < size - 1) {
        row += "  ";
      }
    }
    row += `│${(r + 1).toString().padStart(2)}`;
    console.log(row);
  }

  console.log(borderBottom);
  console.log(top);
}

function getSquareAlg(sq) {
  const cols = "abcdefghijk";
  const row = getRow(sq)
  const col = getCol(sq)
  return `${cols[col]}${row + 1}`;
}

export function getMoveAlg(move: number) {
  return `${getSquareAlg(moveFrom(move))}-${getSquareAlg(moveTo(move))}`;
}
