import type { TemplateRef } from "vue";
export declare const useBoardGeometry: (ref: TemplateRef<HTMLDivElement>) => {
    getCellByAbsolutePosition: (x: number, y: number) => number | null;
    getRelativePosition: (x: number, y: number) => {
        relativeX: number;
        relativeY: number;
    };
};
//# sourceMappingURL=useBoardGeometry.d.ts.map