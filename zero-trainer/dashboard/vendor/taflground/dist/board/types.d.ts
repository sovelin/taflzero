import type { BoardThemeValue } from "./constants";
import type { PieceThemeValue } from "./components/PieceCell";
import { Move, Piece } from "../types";
export interface TaflBoardProps {
    piecesList: Piece[];
    selectableSquares: Set<number>;
    size: number;
    boardTheme: BoardThemeValue;
    pieceTheme: PieceThemeValue;
    highlightMove: Move | null;
    preMove: Move | null;
    possibleMoves: Move[];
}
//# sourceMappingURL=types.d.ts.map