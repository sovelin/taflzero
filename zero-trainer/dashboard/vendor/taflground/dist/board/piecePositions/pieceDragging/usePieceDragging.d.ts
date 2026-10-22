import { type Ref } from "vue";
import { PieceState } from "../../piecePositions/pieceDragging/usePieceDraggingStateMachine";
interface Params {
    boardRef: Ref<HTMLDivElement | null>;
    onDragEnd?: (from: number, to: number) => void;
    onChangePieceState?: (from: PieceState, to: PieceState) => void;
    isPieceSelectable: (sq: number) => boolean;
}
export declare const usePieceDragging: ({ boardRef, onDragEnd, onChangePieceState, isPieceSelectable, }: Params) => {
    movingPiece: import("vue").ComputedRef<{
        sq: number;
        x: number;
        y: number;
        overSquare: number | null;
    } | null>;
    selectedPiece: import("vue").ComputedRef<number | null>;
    overSquare: import("vue").ComputedRef<number | null>;
    cancelPointer: () => void;
};
export {};
//# sourceMappingURL=usePieceDragging.d.ts.map