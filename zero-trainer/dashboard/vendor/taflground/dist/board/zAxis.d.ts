export declare enum ZAxisStates {
    DEFAULT_PIECE = 0,
    ANIMATING_PIECE = 1,
    DRAGGING_PIECE = 2,
    CELL_HIGHLIGHT_DEFAULT = 3,
    CELL_HIGHLIGHT_OVER_PIECE = 4,
    CELL_HIGHLIGHT_POSSIBLE_MOVE = 5
}
export declare const ZAxisValues: {
    0: number;
    1: number;
    2: number;
    3: number;
    4: number;
    5: number;
};
export declare const getPieceZIndex: (isAnimating: boolean, isDragging: boolean) => number;
export declare const getCellHighlighterZIndex: (isOverPiece: boolean) => number;
export declare const getPossibleMoveHighlighterZIndex: () => number;
//# sourceMappingURL=zAxis.d.ts.map