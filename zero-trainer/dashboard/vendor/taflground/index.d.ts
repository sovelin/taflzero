export { default } from "./dist/index";
export * from "./dist/index";

declare module "vue" {
  export interface GlobalComponents {
    TaflBoard: typeof import("./dist/index")["TaflBoard"];
  }
}
