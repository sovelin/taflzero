import { Move } from "../../types";
interface Props {
    highlightMove: Move | null;
    preMove: Move | null;
    movingFrom: number | null;
    overSquare: number | null;
    boardSize: number;
    possibleMovesFromSquare: number[] | null;
}
declare const _default: import("vue").DefineComponent<Props, {}, {}, {}, {}, import("vue").ComponentOptionsMixin, import("vue").ComponentOptionsMixin, {}, string, import("vue").PublicProps, Readonly<Props> & Readonly<{}>, {}, {}, {}, {}, string, import("vue").ComponentProvideOptions, false, {}, any>;
export default _default;
//# sourceMappingURL=HighlightedMoves.vue.d.ts.map