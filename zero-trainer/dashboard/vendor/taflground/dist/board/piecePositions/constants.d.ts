import { computeCellPosition } from "../piecePositions/geometry";
import { PieceType } from "../../types";
export interface PiecePosition {
    piece: PieceType;
    geometry: ReturnType<typeof computeCellPosition>;
    square: number;
    key: string;
    isDragging: boolean;
    isAnimating: boolean;
}
//# sourceMappingURL=constants.d.ts.map