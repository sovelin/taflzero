/* @ts-self-types="./taflzero.d.ts" */

//#region exports

export class Engine {
    constructor() {
        throw new Error('cannot invoke `new` directly');
    }
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
     * Board side of the current variant.
     * @returns {number}
     */
    board_size() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.engineclient_board_size(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {Side | undefined}
     */
    check_terminal_state() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.engineclient_check_terminal_state(this.__wbg_ptr);
        return ret === 2 ? undefined : ret;
    }
    /**
     * @param {string} fen
     * @returns {Side | undefined}
     */
    check_terminal_state_for_fen(fen) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ptr0 = passStringToWasm0(fen, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engineclient_check_terminal_state_for_fen(this.__wbg_ptr, ptr0, len0);
        return ret === 2 ? undefined : ret;
    }
    /**
     * @param {string} mv_str
     * @returns {Move}
     */
    create_move_from_algebraic(mv_str) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ptr0 = passStringToWasm0(mv_str, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engineclient_create_move_from_algebraic(this.__wbg_ptr, ptr0, len0);
        return Move.__wrap(ret);
    }
    /**
     * @returns {Move[]}
     */
    get_available_moves() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.engineclient_get_available_moves(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {number} from
     * @returns {Move[]}
     */
    get_available_moves_from_square(from) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        _assertNum(from);
        const ret = wasm.engineclient_get_available_moves_from_square(this.__wbg_ptr, from);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {any[]}
     */
    get_board_state() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.engineclient_get_board_state(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {string}
     */
    get_board_str() {
        let deferred1_0;
        let deferred1_1;
        try {
            if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
            _assertNum(this.__wbg_ptr);
            const ret = wasm.engineclient_get_board_str(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {string}
     */
    get_fen() {
        let deferred1_0;
        let deferred1_1;
        try {
            if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
            _assertNum(this.__wbg_ptr);
            const ret = wasm.engineclient_get_fen(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {bigint}
     */
    get_zobrist_hash() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.engineclient_get_zobrist_hash(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @param {number} from
     * @param {number} to
     * @returns {boolean}
     */
    is_move_available(from, to) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        _assertNum(from);
        _assertNum(to);
        const ret = wasm.engineclient_is_move_available(this.__wbg_ptr, from, to);
        return ret !== 0;
    }
    /**
     * @param {Move} mv
     */
    make_move(mv) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        _assertClass(mv, Move);
        if (mv.__wbg_ptr === 0) {
            throw new Error('Attempt to use a moved value');
        }
        var ptr0 = mv.__destroy_into_raw();
        wasm.engineclient_make_move(this.__wbg_ptr, ptr0);
    }
    /**
     * @param {number} time
     * @returns {number}
     */
    make_search(time) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        _assertNum(time);
        const ret = wasm.engineclient_make_search(this.__wbg_ptr, time);
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
            if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
            _assertNum(this.__wbg_ptr);
            _assertNum(mv_num);
            const ret = wasm.engineclient_move_num_to_str(this.__wbg_ptr, mv_num);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {string} mv_str
     * @returns {number}
     */
    move_str_to_num(mv_str) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ptr0 = passStringToWasm0(mv_str, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engineclient_move_str_to_num(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    constructor() {
        const ret = wasm.engineclient_new();
        this.__wbg_ptr = ret;
        EngineClientFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {string} fen
     */
    set_fen(fen) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ptr0 = passStringToWasm0(fen, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.engineclient_set_fen(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string} fen
     * @param {Uint32Array} moves
     */
    set_position_and_moves(fen, moves) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ptr0 = passStringToWasm0(fen, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(moves, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        wasm.engineclient_set_position_and_moves(this.__wbg_ptr, ptr0, len0, ptr1, len1);
    }
    /**
     * Fails on an unknown variant instead of silently keeping the old one — a caller
     * that mistypes it would otherwise adjudicate the wrong game.
     * @param {string} variant
     */
    set_variant(variant) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ptr0 = passStringToWasm0(variant, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engineclient_set_variant(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    setup_initial_position() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        wasm.engineclient_setup_initial_position(this.__wbg_ptr);
    }
    /**
     * @returns {Side}
     */
    side_to_move() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.engineclient_side_to_move(this.__wbg_ptr);
        return ret;
    }
    /**
     * Number of squares on the current variant's board.
     * @returns {number}
     */
    total_squares() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.engineclient_total_squares(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) EngineClient.prototype[Symbol.dispose] = EngineClient.prototype.free;

export class Move {
    static __wrap(ptr) {
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
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.move_from(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} mv_u32
     * @returns {Move}
     */
    static from_u32(mv_u32) {
        _assertNum(mv_u32);
        const ret = wasm.move_from_u32(mv_u32);
        return Move.__wrap(ret);
    }
    /**
     * @returns {boolean}
     */
    is_null() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.move_is_null(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {number} from
     * @param {number} to
     */
    constructor(from, to) {
        _assertNum(from);
        _assertNum(to);
        const ret = wasm.move_new(from, to);
        this.__wbg_ptr = ret;
        MoveFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    raw() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.move_raw(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    to() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
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
    constructor() {
        throw new Error('cannot invoke `new` directly');
    }
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
     * Plies to a proven mate (>0 = we mate, <0 = we get mated), else None.
     * @returns {number | undefined}
     */
    get mate() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.__wbg_get_searchiterationresponse_mate(this.__wbg_ptr);
        return ret === Number.MAX_SAFE_INTEGER ? undefined : ret;
    }
    /**
     * @returns {number | undefined}
     */
    get multi_pv() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.__wbg_get_searchiterationresponse_multi_pv(this.__wbg_ptr);
        return ret === Number.MAX_SAFE_INTEGER ? undefined : ret;
    }
    /**
     * @returns {bigint}
     */
    get nodes() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.__wbg_get_searchiterationresponse_nodes(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @returns {number}
     */
    get score() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.__wbg_get_searchiterationresponse_score(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {bigint}
     */
    get speed() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.__wbg_get_searchiterationresponse_speed(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @returns {bigint}
     */
    get time() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.__wbg_get_searchiterationresponse_time(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @returns {number}
     */
    get winrate() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.__wbg_get_searchiterationresponse_winrate(this.__wbg_ptr);
        return ret;
    }
    /**
     * Plies to a proven mate (>0 = we mate, <0 = we get mated), else None.
     * @param {number | null} [arg0]
     */
    set mate(arg0) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        if (!isLikeNone(arg0)) {
            _assertNum(arg0);
        }
        wasm.__wbg_set_searchiterationresponse_mate(this.__wbg_ptr, isLikeNone(arg0) ? Number.MAX_SAFE_INTEGER : (arg0) >> 0);
    }
    /**
     * @param {number | null} [arg0]
     */
    set multi_pv(arg0) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        if (!isLikeNone(arg0)) {
            _assertNum(arg0);
        }
        wasm.__wbg_set_searchiterationresponse_multi_pv(this.__wbg_ptr, isLikeNone(arg0) ? Number.MAX_SAFE_INTEGER : (arg0) >>> 0);
    }
    /**
     * @param {bigint} arg0
     */
    set nodes(arg0) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        _assertBigInt(arg0);
        wasm.__wbg_set_searchiterationresponse_nodes(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set score(arg0) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        _assertNum(arg0);
        wasm.__wbg_set_searchiterationresponse_score(this.__wbg_ptr, arg0);
    }
    /**
     * @param {bigint} arg0
     */
    set speed(arg0) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        _assertBigInt(arg0);
        wasm.__wbg_set_searchiterationresponse_speed(this.__wbg_ptr, arg0);
    }
    /**
     * @param {bigint} arg0
     */
    set time(arg0) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        _assertBigInt(arg0);
        wasm.__wbg_set_searchiterationresponse_time(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set winrate(arg0) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        wasm.__wbg_set_searchiterationresponse_winrate(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) SearchIterationResponse.prototype[Symbol.dispose] = SearchIterationResponse.prototype.free;

export class SearchResponse {
    constructor() {
        throw new Error('cannot invoke `new` directly');
    }
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
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.__wbg_get_searchresponse_best_move(this.__wbg_ptr);
        return Move.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    get score() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.__wbg_get_searchresponse_score(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Move} arg0
     */
    set best_move(arg0) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        _assertClass(arg0, Move);
        if (arg0.__wbg_ptr === 0) {
            throw new Error('Attempt to use a moved value');
        }
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_searchresponse_best_move(this.__wbg_ptr, ptr0);
    }
    /**
     * @param {number} arg0
     */
    set score(arg0) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        _assertNum(arg0);
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
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.timer_elapsed_ms(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    constructor() {
        const ret = wasm.timer_new();
        this.__wbg_ptr = ret;
        TimerFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    start() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
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
     */
    constructor(event_name) {
        const ptr0 = passStringToWasm0(event_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmclient_new(ptr0, len0);
        this.__wbg_ptr = ret;
        WasmClientFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    print_board() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        wasm.wasmclient_print_board(this.__wbg_ptr);
    }
    /**
     * @param {string} cmd
     */
    run(cmd) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ptr0 = passStringToWasm0(cmd, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmclient_run(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Throws on failure: a silently ignored error would leave the engine
     * searching with the previous (or no) network.
     * @param {Uint8Array} data
     */
    set_nn(data) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ptr0 = passArray8ToWasm0(data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmclient_set_nn(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Register the SharedArrayBuffer-backed views used for NN inference done
     * by the onnxruntime-web worker.  Call once after construction, before any
     * search.  See `nn_wasm::set_nn_buffers` for the buffer layout.
     *   control: Int32Array [REQ, RESP, BATCH]
     *   input:   Float32Array  max_batch * 1331   (SAMPLE_SIZE)
     *   output:  Float32Array  max_batch * 4841   (POLICY_SIZE then value)
     * @param {Int32Array} control
     * @param {Float32Array} input
     * @param {Float32Array} output
     */
    set_nn_buffers(control, input, output) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        wasm.wasmclient_set_nn_buffers(this.__wbg_ptr, control, input, output);
    }
    /**
     * Register a SharedArrayBuffer-backed Int32Array as the stop signal.
     * The main thread can stop an ongoing `go infinite` by calling:
     *   `Atomics.store(buffer, 0, 1)`
     * Reset before each new search with `Atomics.store(buffer, 0, 0)`.
     * @param {Int32Array} buffer
     */
    set_stop_buffer(buffer) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        wasm.wasmclient_set_stop_buffer(this.__wbg_ptr, buffer);
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
        const ret = wasm.build_info();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @param {number} sq
 * @param {number} board_size
 * @returns {number}
 */
export function get_col(sq, board_size) {
    _assertNum(sq);
    _assertNum(board_size);
    const ret = wasm.get_col(sq, board_size);
    return ret >>> 0;
}

/**
 * @returns {string}
 */
export function get_initial_board_fen() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_initial_board_fen();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @param {number} sq
 * @param {number} board_size
 * @returns {number}
 */
export function get_row(sq, board_size) {
    _assertNum(sq);
    _assertNum(board_size);
    const ret = wasm.get_row(sq, board_size);
    return ret >>> 0;
}

/**
 * @param {number} sq
 * @param {number} board_size
 * @returns {string}
 */
export function get_sq_algebraic(sq, board_size) {
    let deferred1_0;
    let deferred1_1;
    try {
        _assertNum(sq);
        _assertNum(board_size);
        const ret = wasm.get_sq_algebraic(sq, board_size);
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @param {number} row
 * @param {number} col
 * @param {number} board_size
 * @returns {number}
 */
export function get_square(row, col, board_size) {
    _assertNum(row);
    _assertNum(col);
    _assertNum(board_size);
    const ret = wasm.get_square(row, col, board_size);
    return ret >>> 0;
}

/**
 * @param {string} coord
 * @param {number} board_size
 * @returns {number}
 */
export function get_square_from_algebraic(coord, board_size) {
    const ptr0 = passStringToWasm0(coord, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    _assertNum(board_size);
    const ret = wasm.get_square_from_algebraic(ptr0, len0, board_size);
    return ret >>> 0;
}

/**
 * @returns {string}
 */
export function hello() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.hello();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

export function main_js() {
    wasm.main_js();
}

//#endregion

//#region wasm imports
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_debug_string_8a447059637473e2: function(arg0, arg1) {
            const ret = debugString(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_is_undefined_721f8decd50c87a3: function(arg0) {
            const ret = arg0 === undefined;
            _assertBoolean(ret);
            return ret;
        },
        __wbg___wbindgen_throw_ea4887a5f8f9a9db: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_dispatchEvent_ba2bd0c05d922867: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.dispatchEvent(arg1);
            _assertBoolean(ret);
            return ret;
        }, arguments); },
        __wbg_error_a6fa202b58aa1cd3: function() { return logError(function (arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        }, arguments); },
        __wbg_instanceof_EventTarget_72098ffe4ac5efa2: function() { return logError(function (arg0) {
            let result;
            try {
                result = arg0 instanceof EventTarget;
            } catch (_) {
                result = false;
            }
            const ret = result;
            _assertBoolean(ret);
            return ret;
        }, arguments); },
        __wbg_length_e6bdba0734e5c7f5: function() { return logError(function (arg0) {
            const ret = arg0.length;
            _assertNum(ret);
            return ret;
        }, arguments); },
        __wbg_load_092326361fd87ff1: function() { return handleError(function (arg0, arg1) {
            const ret = Atomics.load(arg0, arg1 >>> 0);
            _assertNum(ret);
            return ret;
        }, arguments); },
        __wbg_load_452389fa0d16b447: function() { return handleError(function (arg0, arg1) {
            const ret = Atomics.load(arg0, arg1 >>> 0);
            _assertNum(ret);
            return ret;
        }, arguments); },
        __wbg_move_new: function() { return logError(function (arg0) {
            const ret = Move.__wrap(arg0);
            return ret;
        }, arguments); },
        __wbg_new_227d7c05414eb861: function() { return logError(function () {
            const ret = new Error();
            return ret;
        }, arguments); },
        __wbg_new_2e117a478906f062: function() { return logError(function () {
            const ret = new Object();
            return ret;
        }, arguments); },
        __wbg_new_with_event_init_dict_6c2c9e8388908eb7: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = new CustomEvent(getStringFromWasm0(arg0, arg1), arg2);
            return ret;
        }, arguments); },
        __wbg_notify_cefc0187827ce499: function() { return handleError(function (arg0, arg1) {
            const ret = Atomics.notify(arg0, arg1 >>> 0);
            _assertNum(ret);
            return ret;
        }, arguments); },
        __wbg_now_d2e0afbad4edbe82: function() { return logError(function () {
            const ret = Date.now();
            return ret;
        }, arguments); },
        __wbg_prototypesetcall_46d9a0b93af9df2d: function() { return logError(function (arg0, arg1, arg2) {
            Float32Array.prototype.set.call(getArrayF32FromWasm0(arg0, arg1), arg2);
        }, arguments); },
        __wbg_set_6f5ddc74972bd54e: function() { return logError(function (arg0, arg1, arg2) {
            arg0.set(getArrayF32FromWasm0(arg1, arg2));
        }, arguments); },
        __wbg_set_detail_016b98403f732421: function() { return logError(function (arg0, arg1) {
            arg0.detail = arg1;
        }, arguments); },
        __wbg_stack_3b0d974bbf31e44f: function() { return logError(function (arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        }, arguments); },
        __wbg_static_accessor_GLOBAL_THIS_2fee5048bcca5938: function() { return logError(function () {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        }, arguments); },
        __wbg_static_accessor_GLOBAL_ce44e66a4935da8c: function() { return logError(function () {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        }, arguments); },
        __wbg_static_accessor_SELF_44f6e0cb5e67cdad: function() { return logError(function () {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        }, arguments); },
        __wbg_static_accessor_WINDOW_168f178805d978fe: function() { return logError(function () {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        }, arguments); },
        __wbg_store_26c6bfd2829c839e: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Atomics.store(arg0, arg1 >>> 0, arg2);
            _assertNum(ret);
            return ret;
        }, arguments); },
        __wbg_subarray_7628896583dc2831: function() { return logError(function (arg0, arg1, arg2) {
            const ret = arg0.subarray(arg1 >>> 0, arg2 >>> 0);
            return ret;
        }, arguments); },
        __wbg_wait_884ad5cc4da58248: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Atomics.wait(arg0, arg1 >>> 0, arg2);
            return ret;
        }, arguments); },
        __wbindgen_cast_0000000000000001: function() { return logError(function (arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        }, arguments); },
        __wbindgen_cast_0000000000000002: function() { return logError(function (arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        }, arguments); },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./taflzero_bg.js": import0,
    };
}


//#endregion
const EngineFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_engine_free(ptr, 1));
const EngineClientFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_engineclient_free(ptr, 1));
const MoveFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_move_free(ptr, 1));
const SearchIterationResponseFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_searchiterationresponse_free(ptr, 1));
const SearchResponseFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_searchresponse_free(ptr, 1));
const TimerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_timer_free(ptr, 1));
const WasmClientFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmclient_free(ptr, 1));


//#region intrinsics
function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

function _assertBigInt(n) {
    if (typeof(n) !== 'bigint') throw new Error(`expected a bigint argument, found ${typeof(n)}`);
}

function _assertBoolean(n) {
    if (typeof(n) !== 'boolean') {
        throw new Error(`expected a boolean argument, found ${typeof(n)}`);
    }
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function _assertNum(n) {
    if (typeof(n) !== 'number') throw new Error(`expected a number argument, found ${typeof(n)}`);
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

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayJsValueFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    const mem = getDataViewMemory0();
    const result = [];
    for (let i = ptr; i < ptr + 4 * len; i += 4) {
        result.push(wasm.__wbindgen_externrefs.get(mem.getUint32(i, true)));
    }
    wasm.__externref_drop_slice(ptr, len);
    return result;
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
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

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function logError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        let error = (function () {
            try {
                return e instanceof Error ? `${e.message}\n\nStack:\n${e.stack}` : e.toString();
            } catch(_) {
                return "<failed to stringify thrown value>";
            }
        }());
        console.error("wasm-bindgen: imported JS function that was not marked as `catch` threw an error:", error);
        throw e;
    }
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (typeof(arg) !== 'string') throw new Error(`expected a string argument, found ${typeof(arg)}`);
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
        if (ret.read !== arg.length) throw new Error('failed to pass whole string');
        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
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


//#endregion

//#region wasm loading
let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
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
//#endregion
export { wasm as __wasm }
