import type { Coords } from "./types";
export declare enum PieceState {
    IDLE = "IDLE",
    PRESSED = "PRESSED",
    PRESSED_DOUBLE = "PRESSED_DOUBLE",
    SELECTED = "SELECTED",
    MOVING = "MOVING",
    PRE_CLICKING = "PRE_CLICKING"
}
interface Params {
    isPieceSelectable: (sq: number) => boolean;
    onMove?: (from: number, to: number) => void;
    onChangeState?: (from: PieceState, to: PieceState) => void;
}
export declare const usePieceDraggingStateMachine: ({ isPieceSelectable, onMove, onChangeState }: Params) => {
    actions: {
        pointerDown: (sq: number | null, coords: Coords) => void;
        pointerMove: (coords: Coords) => void;
        pointerUp: (sq: number | null) => void;
        pointerCancel: () => void;
    };
    selectedPiece: import("vue").ComputedRef<{
        sq: number;
        isMoving?: undefined;
    } | {
        sq: number;
        isMoving: {
            x: number;
            y: number;
        };
    } | null>;
    isMoving: import("vue").ComputedRef<boolean>;
    coords: import("vue").ComputedRef<{
        x: number;
        y: number;
    }>;
};
export {};
//# sourceMappingURL=usePieceDraggingStateMachine.d.ts.map