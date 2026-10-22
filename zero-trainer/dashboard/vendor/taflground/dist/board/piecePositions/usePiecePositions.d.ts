import { type Ref } from "vue";
import type { PieceState } from "../piecePositions/pieceDragging/usePieceDraggingStateMachine";
import { Piece, PieceType } from "../../types";
interface Params {
    boardPieces: Ref<Piece[]>;
    boardRef: Ref<HTMLDivElement | null>;
    onDragEnd?: (from: number, to: number) => void;
    onChangePieceState?: (from: PieceState, to: PieceState) => void;
    isPieceSelectable: (sq: number) => boolean;
}
export declare const usePiecePositions: ({ boardPieces, boardRef, onDragEnd, onChangePieceState, isPieceSelectable }: Params) => {
    piecePositions: import("vue").ComputedRef<{
        piece: PieceType;
        geometry: {
            left: number;
            top: number;
        };
        square: number;
        key: string;
        isDragging: boolean;
        isAnimating: boolean;
    }[]>;
    selectedPiece: import("vue").ComputedRef<number | null>;
    overSquare: import("vue").ComputedRef<number | null>;
};
export {};
//# sourceMappingURL=usePiecePositions.d.ts.map