/* tslint:disable */
/* eslint-disable */
export function main_js(): void;
export function build_info(): string;
export function hello(): string;
export function get_initial_board_fen(): string;
export function get_total_squares(): number;
export function get_board_size(): number;
export function get_square_from_algebraic(coord: string): number;
export function get_row(sq: number): number;
export function get_square(row: number, col: number): number;
export function get_col(sq: number): number;
export function get_sq_algebraic(sq: number): string;
export enum Piece {
  EMPTY = 0,
  ATTACKER = 1,
  DEFENDER = 2,
  KING = 3,
}
export enum Side {
  ATTACKERS = 0,
  DEFENDERS = 1,
}
export class Engine {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
}
export class EngineClient {
  free(): void;
  [Symbol.dispose](): void;
  make_search(time: number, depth: number): number;
  get_w2_first(): number;
  side_to_move(): Side;
  get_board_str(): string;
  get_board_state(): any[];
  move_num_to_str(mv_num: number): string;
  move_str_to_num(mv_str: string): number;
  get_zobrist_hash(): bigint;
  is_move_available(from: number, to: number): boolean;
  get_available_moves(from: number): Move[];
  check_terminal_state(): Side | undefined;
  set_position_and_moves(fen: string, moves: Uint32Array): void;
  create_move_from_algebraic(mv_str: string): Move;
  check_terminal_state_for_fen(fen: string): Side | undefined;
  get_available_moves_from_square(from: number): Move[];
  constructor(tt_size_mb: number);
  get_fen(): string;
  set_fen(fen: string): void;
  make_move(mv: Move): void;
}
export class Move {
  free(): void;
  [Symbol.dispose](): void;
  static create_null(): Move;
  to(): number;
  constructor(from: number, to: number);
  raw(): number;
  from(): number;
  is_null(): boolean;
  static from_u32(mv_u32: number): Move;
}
export class SearchIterationResponse {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  depth: number;
  mv: Move;
  score: number;
  nodes: bigint;
  time: bigint;
  speed: bigint;
}
export class SearchResponse {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  best_move: Move;
  score: number;
}
export class Timer {
  free(): void;
  [Symbol.dispose](): void;
  elapsed_ms(): bigint;
  constructor();
  start(): void;
}
export class WasmClient {
  free(): void;
  [Symbol.dispose](): void;
  print_board(): void;
  constructor(event_name: string, tt_size: number);
  run(cmd: string): void;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_engine_free: (a: number, b: number) => void;
  readonly __wbg_engineclient_free: (a: number, b: number) => void;
  readonly __wbg_get_searchiterationresponse_depth: (a: number) => number;
  readonly __wbg_get_searchiterationresponse_mv: (a: number) => number;
  readonly __wbg_get_searchiterationresponse_nodes: (a: number) => bigint;
  readonly __wbg_get_searchiterationresponse_score: (a: number) => number;
  readonly __wbg_get_searchiterationresponse_speed: (a: number) => bigint;
  readonly __wbg_get_searchiterationresponse_time: (a: number) => bigint;
  readonly __wbg_get_searchresponse_best_move: (a: number) => number;
  readonly __wbg_get_searchresponse_score: (a: number) => number;
  readonly __wbg_move_free: (a: number, b: number) => void;
  readonly __wbg_searchiterationresponse_free: (a: number, b: number) => void;
  readonly __wbg_searchresponse_free: (a: number, b: number) => void;
  readonly __wbg_set_searchiterationresponse_depth: (a: number, b: number) => void;
  readonly __wbg_set_searchiterationresponse_mv: (a: number, b: number) => void;
  readonly __wbg_set_searchiterationresponse_nodes: (a: number, b: bigint) => void;
  readonly __wbg_set_searchiterationresponse_score: (a: number, b: number) => void;
  readonly __wbg_set_searchiterationresponse_speed: (a: number, b: bigint) => void;
  readonly __wbg_set_searchiterationresponse_time: (a: number, b: bigint) => void;
  readonly __wbg_set_searchresponse_best_move: (a: number, b: number) => void;
  readonly __wbg_set_searchresponse_score: (a: number, b: number) => void;
  readonly __wbg_timer_free: (a: number, b: number) => void;
  readonly __wbg_wasmclient_free: (a: number, b: number) => void;
  readonly build_info: (a: number) => void;
  readonly engineclient_check_terminal_state: (a: number) => number;
  readonly engineclient_check_terminal_state_for_fen: (a: number, b: number, c: number) => number;
  readonly engineclient_create_move_from_algebraic: (a: number, b: number, c: number) => number;
  readonly engineclient_get_available_moves: (a: number, b: number, c: number) => void;
  readonly engineclient_get_available_moves_from_square: (a: number, b: number, c: number) => void;
  readonly engineclient_get_board_state: (a: number, b: number) => void;
  readonly engineclient_get_board_str: (a: number, b: number) => void;
  readonly engineclient_get_fen: (a: number, b: number) => void;
  readonly engineclient_get_w2_first: (a: number) => number;
  readonly engineclient_get_zobrist_hash: (a: number) => bigint;
  readonly engineclient_is_move_available: (a: number, b: number, c: number) => number;
  readonly engineclient_make_move: (a: number, b: number) => void;
  readonly engineclient_make_search: (a: number, b: number, c: number) => number;
  readonly engineclient_move_num_to_str: (a: number, b: number, c: number) => void;
  readonly engineclient_move_str_to_num: (a: number, b: number, c: number, d: number) => void;
  readonly engineclient_new: (a: number) => number;
  readonly engineclient_set_fen: (a: number, b: number, c: number) => void;
  readonly engineclient_set_position_and_moves: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly engineclient_side_to_move: (a: number) => number;
  readonly get_board_size: () => number;
  readonly get_col: (a: number) => number;
  readonly get_initial_board_fen: (a: number) => void;
  readonly get_row: (a: number) => number;
  readonly get_sq_algebraic: (a: number, b: number) => void;
  readonly get_square: (a: number, b: number) => number;
  readonly get_square_from_algebraic: (a: number, b: number) => number;
  readonly get_total_squares: () => number;
  readonly hello: (a: number) => void;
  readonly move_create_null: () => number;
  readonly move_from: (a: number) => number;
  readonly move_from_u32: (a: number) => number;
  readonly move_is_null: (a: number) => number;
  readonly move_new: (a: number, b: number) => number;
  readonly move_raw: (a: number) => number;
  readonly move_to: (a: number) => number;
  readonly timer_elapsed_ms: (a: number) => bigint;
  readonly timer_new: () => number;
  readonly timer_start: (a: number) => void;
  readonly wasmclient_new: (a: number, b: number, c: number) => number;
  readonly wasmclient_print_board: (a: number) => void;
  readonly wasmclient_run: (a: number, b: number, c: number) => void;
  readonly main_js: () => void;
  readonly __wbindgen_export: (a: number, b: number) => number;
  readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_export3: (a: number) => void;
  readonly __wbindgen_export4: (a: number, b: number, c: number) => void;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
