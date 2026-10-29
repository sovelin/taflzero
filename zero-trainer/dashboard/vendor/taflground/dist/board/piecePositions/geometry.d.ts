export declare const Geometry: {
    BOARD_SIZE: number;
    GAP: number;
    CELL_SIZE: number;
};
export declare const gapSizePercent: number;
export declare const cellSizePercent: number;
/**
 * Returns axis position in percent
 */
export declare const computeAxisPosition: (index: number) => number;
/**
 * Returns relative piece position in percent on square (sq)
 * @param sq
 */
export declare const computeCellPosition: (sq: number) => {
    left: number;
    top: number;
};
/**
 * Return relative bounds in percents
 */
export declare const computeCellInPosition: (x: number, y: number) => number | null;
//# sourceMappingURL=geometry.d.ts.map