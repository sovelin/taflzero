export interface CellHighlighterProps {
    sq: number;
    type: 'fill' | 'border' | 'circle';
    color: string;
    opacity: number;
    boardSize: number;
    style?: Record<string, unknown>;
}
declare const _default: import("vue").DefineComponent<CellHighlighterProps, {}, {}, {}, {}, import("vue").ComponentOptionsMixin, import("vue").ComponentOptionsMixin, {}, string, import("vue").PublicProps, Readonly<CellHighlighterProps> & Readonly<{}>, {}, {}, {}, {}, string, import("vue").ComponentProvideOptions, false, {}, any>;
export default _default;
//# sourceMappingURL=CellHighlighter.vue.d.ts.map