/* @ts-self-types="./taflzero.d.ts" */

export class Engine {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EngineFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_engine_free(ptr, 0);
    }
}
if (Symbol.dispose) Engine.prototype[Symbol.dispose] = Engine.prototype.free;

export class EngineClient {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EngineClientFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_engineclient_free(ptr, 0);
    }
    /**
     * @returns {Side | undefined}
     */
    check_terminal_state() {
        const ret = wasm.engineclient_check_terminal_state(this.__wbg_ptr);
        return ret === 2 ? undefined : ret;
    }
    /**
     * @param {string} fen
     * @returns {Side | undefined}
     */
    check_terminal_state_for_fen(fen) {
        const ptr0 = passStringToWasm0(fen, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engineclient_check_terminal_state_for_fen(this.__wbg_ptr, ptr0, len0);
        return ret === 2 ? undefined : ret;
    }
    /**
     * @param {string} mv_str
     * @returns {Move}
     */
    create_move_from_algebraic(mv_str) {
        const ptr0 = passStringToWasm0(mv_str, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engineclient_create_move_from_algebraic(this.__wbg_ptr, ptr0, len0);
        return Move.__wrap(ret);
    }
    /**
     * @param {number} from
     * @returns {Move[]}
     */
    get_available_moves(from) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.engineclient_get_available_moves(retptr, this.__wbg_ptr, from);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayJsValueFromWasm0(r0, r1).slice();
            wasm.__wbindgen_export4(r0, r1 * 4, 4);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {number} from
     * @returns {Move[]}
     */
    get_available_moves_from_square(from) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.engineclient_get_available_moves_from_square(retptr, this.__wbg_ptr, from);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayJsValueFromWasm0(r0, r1).slice();
            wasm.__wbindgen_export4(r0, r1 * 4, 4);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {any[]}
     */
    get_board_state() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.engineclient_get_board_state(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayJsValueFromWasm0(r0, r1).slice();
            wasm.__wbindgen_export4(r0, r1 * 4, 4);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {string}
     */
    get_board_str() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.engineclient_get_board_str(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export4(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {string}
     */
    get_fen() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.engineclient_get_fen(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export4(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {number}
     */
    get_w2_first() {
        const ret = wasm.engineclient_get_w2_first(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {bigint}
     */
    get_zobrist_hash() {
        const ret = wasm.engineclient_get_zobrist_hash(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @param {number} from
     * @param {number} to
     * @returns {boolean}
     */
    is_move_available(from, to) {
        const ret = wasm.engineclient_is_move_available(this.__wbg_ptr, from, to);
        return ret !== 0;
    }
    /**
     * @param {Move} mv
     */
    make_move(mv) {
        _assertClass(mv, Move);
        var ptr0 = mv.__destroy_into_raw();
        wasm.engineclient_make_move(this.__wbg_ptr, ptr0);
    }
    /**
     * @param {number} time
     * @param {number} depth
     * @returns {number}
     */
    make_search(time, depth) {
        const ret = wasm.engineclient_make_search(this.__wbg_ptr, time, depth);
        return ret >>> 0;
    }
    /**
     * @param {number} mv_num
     * @returns {string}
     */
    move_num_to_str(mv_num) {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.engineclient_move_num_to_str(retptr, this.__wbg_ptr, mv_num);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export4(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {string} mv_str
     * @returns {number}
     */
    move_str_to_num(mv_str) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(mv_str, wasm.__wbindgen_export, wasm.__wbindgen_export2);
            const len0 = WASM_VECTOR_LEN;
            wasm.engineclient_move_str_to_num(retptr, this.__wbg_ptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return r0 >>> 0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {number} tt_size_mb
     */
    constructor(tt_size_mb) {
        const ret = wasm.engineclient_new(tt_size_mb);
        this.__wbg_ptr = ret >>> 0;
        EngineClientFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {string} fen
     */
    set_fen(fen) {
        const ptr0 = passStringToWasm0(fen, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len0 = WASM_VECTOR_LEN;
        wasm.engineclient_set_fen(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string} fen
     * @param {Uint32Array} moves
     */
    set_position_and_moves(fen, moves) {
        const ptr0 = passStringToWasm0(fen, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(moves, wasm.__wbindgen_export);
        const len1 = WASM_VECTOR_LEN;
        wasm.engineclient_set_position_and_moves(this.__wbg_ptr, ptr0, len0, ptr1, len1);
    }
    /**
     * @returns {Side}
     */
    side_to_move() {
        const ret = wasm.engineclient_side_to_move(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) EngineClient.prototype[Symbol.dispose] = EngineClient.prototype.free;

export class Move {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Move.prototype);
        obj.__wbg_ptr = ptr;
        MoveFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        MoveFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_move_free(ptr, 0);
    }
    /**
     * @returns {Move}
     */
    static create_null() {
        const ret = wasm.move_create_null();
        return Move.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    from() {
        const ret = wasm.move_from(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} mv_u32
     * @returns {Move}
     */
    static from_u32(mv_u32) {
        const ret = wasm.move_from_u32(mv_u32);
        return Move.__wrap(ret);
    }
    /**
     * @returns {boolean}
     */
    is_null() {
        const ret = wasm.move_is_null(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {number} from
     * @param {number} to
     */
    constructor(from, to) {
        const ret = wasm.move_new(from, to);
        this.__wbg_ptr = ret >>> 0;
        MoveFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    raw() {
        const ret = wasm.move_raw(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    to() {
        const ret = wasm.move_to(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) Move.prototype[Symbol.dispose] = Move.prototype.free;

/**
 * @enum {0 | 1 | 2 | 3}
 */
export const Piece = Object.freeze({
    EMPTY: 0, "0": "EMPTY",
    ATTACKER: 1, "1": "ATTACKER",
    DEFENDER: 2, "2": "DEFENDER",
    KING: 3, "3": "KING",
});

export class SearchIterationResponse {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SearchIterationResponseFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_searchiterationresponse_free(ptr, 0);
    }
    /**
     * @returns {bigint}
     */
    get nodes() {
        const ret = wasm.__wbg_get_searchiterationresponse_nodes(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @returns {number}
     */
    get score() {
        const ret = wasm.__wbg_get_searchiterationresponse_score(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {bigint}
     */
    get speed() {
        const ret = wasm.__wbg_get_searchiterationresponse_speed(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @returns {bigint}
     */
    get time() {
        const ret = wasm.__wbg_get_searchiterationresponse_time(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @returns {number}
     */
    get winrate() {
        const ret = wasm.__wbg_get_searchiterationresponse_winrate(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {bigint} arg0
     */
    set nodes(arg0) {
        wasm.__wbg_set_searchiterationresponse_nodes(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set score(arg0) {
        wasm.__wbg_set_searchiterationresponse_score(this.__wbg_ptr, arg0);
    }
    /**
     * @param {bigint} arg0
     */
    set speed(arg0) {
        wasm.__wbg_set_searchiterationresponse_speed(this.__wbg_ptr, arg0);
    }
    /**
     * @param {bigint} arg0
     */
    set time(arg0) {
        wasm.__wbg_set_searchiterationresponse_time(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set winrate(arg0) {
        wasm.__wbg_set_searchiterationresponse_winrate(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) SearchIterationResponse.prototype[Symbol.dispose] = SearchIterationResponse.prototype.free;

export class SearchResponse {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SearchResponseFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_searchresponse_free(ptr, 0);
    }
    /**
     * @returns {Move}
     */
    get best_move() {
        const ret = wasm.__wbg_get_searchresponse_best_move(this.__wbg_ptr);
        return Move.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    get score() {
        const ret = wasm.__wbg_get_searchresponse_score(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Move} arg0
     */
    set best_move(arg0) {
        _assertClass(arg0, Move);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_searchresponse_best_move(this.__wbg_ptr, ptr0);
    }
    /**
     * @param {number} arg0
     */
    set score(arg0) {
        wasm.__wbg_set_searchresponse_score(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) SearchResponse.prototype[Symbol.dispose] = SearchResponse.prototype.free;

/**
 * @enum {0 | 1}
 */
export const Side = Object.freeze({
    ATTACKERS: 0, "0": "ATTACKERS",
    DEFENDERS: 1, "1": "DEFENDERS",
});

export class Timer {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TimerFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_timer_free(ptr, 0);
    }
    /**
     * @returns {bigint}
     */
    elapsed_ms() {
        const ret = wasm.timer_elapsed_ms(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    constructor() {
        const ret = wasm.timer_new();
        this.__wbg_ptr = ret >>> 0;
        TimerFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    start() {
        wasm.timer_start(this.__wbg_ptr);
    }
}
if (Symbol.dispose) Timer.prototype[Symbol.dispose] = Timer.prototype.free;

export class WasmClient {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmClientFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmclient_free(ptr, 0);
    }
    /**
     * @param {string} event_name
     * @param {number} tt_size
     */
    constructor(event_name, tt_size) {
        const ptr0 = passStringToWasm0(event_name, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmclient_new(ptr0, len0, tt_size);
        this.__wbg_ptr = ret >>> 0;
        WasmClientFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    print_board() {
        wasm.wasmclient_print_board(this.__wbg_ptr);
    }
    /**
     * @param {string} cmd
     */
    run(cmd) {
        const ptr0 = passStringToWasm0(cmd, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmclient_run(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Register a SharedArrayBuffer-backed Int32Array as the stop signal.
     * The main thread can stop an ongoing `go infinite` by calling:
     *   `Atomics.store(buffer, 0, 1)`
     * Reset before each new search with `Atomics.store(buffer, 0, 0)`.
     * @param {Int32Array} buffer
     */
    set_stop_buffer(buffer) {
        wasm.wasmclient_set_stop_buffer(this.__wbg_ptr, addHeapObject(buffer));
    }
}
if (Symbol.dispose) WasmClient.prototype[Symbol.dispose] = WasmClient.prototype.free;

/**
 * @returns {string}
 */
export function build_info() {
    let deferred1_0;
    let deferred1_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.build_info(retptr);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        deferred1_0 = r0;
        deferred1_1 = r1;
        return getStringFromWasm0(r0, r1);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export4(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @returns {number}
 */
export function get_board_size() {
    const ret = wasm.get_board_size();
    return ret >>> 0;
}

/**
 * @param {number} sq
 * @returns {number}
 */
export function get_col(sq) {
    const ret = wasm.get_col(sq);
    return ret >>> 0;
}

/**
 * @returns {string}
 */
export function get_initial_board_fen() {
    let deferred1_0;
    let deferred1_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.get_initial_board_fen(retptr);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        deferred1_0 = r0;
        deferred1_1 = r1;
        return getStringFromWasm0(r0, r1);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export4(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @param {number} sq
 * @returns {number}
 */
export function get_row(sq) {
    const ret = wasm.get_row(sq);
    return ret >>> 0;
}

/**
 * @param {number} sq
 * @returns {string}
 */
export function get_sq_algebraic(sq) {
    let deferred1_0;
    let deferred1_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.get_sq_algebraic(retptr, sq);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        deferred1_0 = r0;
        deferred1_1 = r1;
        return getStringFromWasm0(r0, r1);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export4(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @param {number} row
 * @param {number} col
 * @returns {number}
 */
export function get_square(row, col) {
    const ret = wasm.get_square(row, col);
    return ret >>> 0;
}

/**
 * @param {string} coord
 * @returns {number}
 */
export function get_square_from_algebraic(coord) {
    const ptr0 = passStringToWasm0(coord, wasm.__wbindgen_export, wasm.__wbindgen_export2);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.get_square_from_algebraic(ptr0, len0);
    return ret >>> 0;
}

/**
 * @returns {number}
 */
export function get_total_squares() {
    const ret = wasm.get_total_squares();
    return ret >>> 0;
}

/**
 * @returns {string}
 */
export function hello() {
    let deferred1_0;
    let deferred1_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.hello(retptr);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        deferred1_0 = r0;
        deferred1_1 = r1;
        return getStringFromWasm0(r0, r1);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export4(deferred1_0, deferred1_1, 1);
    }
}

export function main_js() {
    wasm.main_js();
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_debug_string_ab4b34d23d6778bd: function(arg0, arg1) {
            const ret = debugString(getObject(arg1));
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_export, wasm.__wbindgen_export2);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_is_function_3baa9db1a987f47d: function(arg0) {
            const ret = typeof(getObject(arg0)) === 'function';
            return ret;
        },
        __wbg___wbindgen_is_object_63322ec0cd6ea4ef: function(arg0) {
            const val = getObject(arg0);
            const ret = typeof(val) === 'object' && val !== null;
            return ret;
        },
        __wbg___wbindgen_is_string_6df3bf7ef1164ed3: function(arg0) {
            const ret = typeof(getObject(arg0)) === 'string';
            return ret;
        },
        __wbg___wbindgen_is_undefined_29a43b4d42920abd: function(arg0) {
            const ret = getObject(arg0) === undefined;
            return ret;
        },
        __wbg___wbindgen_throw_6b64449b9b9ed33c: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_call_a24592a6f349a97e: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = getObject(arg0).call(getObject(arg1), getObject(arg2));
            return addHeapObject(ret);
        }, arguments); },
        __wbg_crypto_38df2bab126b63dc: function(arg0) {
            const ret = getObject(arg0).crypto;
            return addHeapObject(ret);
        },
        __wbg_dispatchEvent_d8729fc7c0462621: function() { return handleError(function (arg0, arg1) {
            const ret = getObject(arg0).dispatchEvent(getObject(arg1));
            return ret;
        }, arguments); },
        __wbg_error_a6fa202b58aa1cd3: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_export4(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_getRandomValues_c44a50d8cfdaebeb: function() { return handleError(function (arg0, arg1) {
            getObject(arg0).getRandomValues(getObject(arg1));
        }, arguments); },
        __wbg_instanceof_EventTarget_8669c4aa1d0ee593: function(arg0) {
            let result;
            try {
                result = getObject(arg0) instanceof EventTarget;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_length_9f1775224cf1d815: function(arg0) {
            const ret = getObject(arg0).length;
            return ret;
        },
        __wbg_load_452389fa0d16b447: function() { return handleError(function (arg0, arg1) {
            const ret = Atomics.load(getObject(arg0), arg1 >>> 0);
            return ret;
        }, arguments); },
        __wbg_move_new: function(arg0) {
            const ret = Move.__wrap(arg0);
            return addHeapObject(ret);
        },
        __wbg_msCrypto_bd5a034af96bcba6: function(arg0) {
            const ret = getObject(arg0).msCrypto;
            return addHeapObject(ret);
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return addHeapObject(ret);
        },
        __wbg_new_aa8d0fa9762c29bd: function() {
            const ret = new Object();
            return addHeapObject(ret);
        },
        __wbg_new_with_event_init_dict_5f4f1e5009457f42: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = new CustomEvent(getStringFromWasm0(arg0, arg1), getObject(arg2));
            return addHeapObject(ret);
        }, arguments); },
        __wbg_new_with_length_8c854e41ea4dae9b: function(arg0) {
            const ret = new Uint8Array(arg0 >>> 0);
            return addHeapObject(ret);
        },
        __wbg_node_84ea875411254db1: function(arg0) {
            const ret = getObject(arg0).node;
            return addHeapObject(ret);
        },
        __wbg_now_a9b7df1cbee90986: function() {
            const ret = Date.now();
            return ret;
        },
        __wbg_process_44c7a14e11e9f69e: function(arg0) {
            const ret = getObject(arg0).process;
            return addHeapObject(ret);
        },
        __wbg_prototypesetcall_a6b02eb00b0f4ce2: function(arg0, arg1, arg2) {
            Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), getObject(arg2));
        },
        __wbg_randomFillSync_6c25eac9869eb53c: function() { return handleError(function (arg0, arg1) {
            getObject(arg0).randomFillSync(takeObject(arg1));
        }, arguments); },
        __wbg_require_b4edbdcf3e2a1ef0: function() { return handleError(function () {
            const ret = module.require;
            return addHeapObject(ret);
        }, arguments); },
        __wbg_set_detail_68bec5c91196ba57: function(arg0, arg1) {
            getObject(arg0).detail = getObject(arg1);
        },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = getObject(arg1).stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_export, wasm.__wbindgen_export2);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_static_accessor_GLOBAL_8cfadc87a297ca02: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addHeapObject(ret);
        },
        __wbg_static_accessor_GLOBAL_THIS_602256ae5c8f42cf: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addHeapObject(ret);
        },
        __wbg_static_accessor_SELF_e445c1c7484aecc3: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addHeapObject(ret);
        },
        __wbg_static_accessor_WINDOW_f20e8576ef1e0f17: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addHeapObject(ret);
        },
        __wbg_subarray_f8ca46a25b1f5e0d: function(arg0, arg1, arg2) {
            const ret = getObject(arg0).subarray(arg1 >>> 0, arg2 >>> 0);
            return addHeapObject(ret);
        },
        __wbg_versions_276b2795b1c6a219: function(arg0) {
            const ret = getObject(arg0).versions;
            return addHeapObject(ret);
        },
        __wbindgen_cast_0000000000000001: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return addHeapObject(ret);
        },
        __wbindgen_cast_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(U8)) -> NamedExternref("Uint8Array")`.
            const ret = getArrayU8FromWasm0(arg0, arg1);
            return addHeapObject(ret);
        },
        __wbindgen_cast_0000000000000003: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return addHeapObject(ret);
        },
        __wbindgen_object_clone_ref: function(arg0) {
            const ret = getObject(arg0);
            return addHeapObject(ret);
        },
        __wbindgen_object_drop_ref: function(arg0) {
            takeObject(arg0);
        },
    };
    return {
        __proto__: null,
        "./taflzero_bg.js": import0,
    };
}

const EngineFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_engine_free(ptr >>> 0, 1));
const EngineClientFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_engineclient_free(ptr >>> 0, 1));
const MoveFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_move_free(ptr >>> 0, 1));
const SearchIterationResponseFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_searchiterationresponse_free(ptr >>> 0, 1));
const SearchResponseFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_searchresponse_free(ptr >>> 0, 1));
const TimerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_timer_free(ptr >>> 0, 1));
const WasmClientFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmclient_free(ptr >>> 0, 1));

function addHeapObject(obj) {
    if (heap_next === heap.length) heap.push(heap.length + 1);
    const idx = heap_next;
    heap_next = heap[idx];

    heap[idx] = obj;
    return idx;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

function dropObject(idx) {
    if (idx < 1028) return;
    heap[idx] = heap_next;
    heap_next = idx;
}

function getArrayJsValueFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    const mem = getDataViewMemory0();
    const result = [];
    for (let i = ptr; i < ptr + 4 * len; i += 4) {
        result.push(takeObject(mem.getUint32(i, true)));
    }
    return result;
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function getObject(idx) { return heap[idx]; }

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        wasm.__wbindgen_export3(addHeapObject(e));
    }
}

let heap = new Array(1024).fill(undefined);
heap.push(undefined, null, true, false);

let heap_next = heap.length;

function isLikeNone(x) {
    return x === undefined || x === null;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeObject(idx) {
    const ret = getObject(idx);
    dropObject(idx);
    return ret;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('taflzero_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
