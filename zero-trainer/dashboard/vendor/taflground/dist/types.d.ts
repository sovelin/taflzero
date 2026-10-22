export interface Move {
    from: number;
    to: number;
}
export type PieceType = 'king' | 'attacker' | 'defender';
export interface Piece {
    type: PieceType;
    sq: number;
}
export type EnumLike<T, K extends keyof T = keyof T> = T[K];
//# sourceMappingURL=types.d.ts.map