let wasm;

let heap = new Array(128).fill(undefined);

heap.push(undefined, null, true, false);

function getObject(idx) { return heap[idx]; }

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

let WASM_VECTOR_LEN = 0;

let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
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
    }
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

let cachedDataViewMemory0 = null;

function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
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

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let heap_next = heap.length;

function addHeapObject(obj) {
    if (heap_next === heap.length) heap.push(heap.length + 1);
    const idx = heap_next;
    heap_next = heap[idx];

    heap[idx] = obj;
    return idx;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        wasm.__wbindgen_export3(addHeapObject(e));
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function dropObject(idx) {
    if (idx < 132) return;
    heap[idx] = heap_next;
    heap_next = idx;
}

function takeObject(idx) {
    const ret = getObject(idx);
    dropObject(idx);
    return ret;
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

export function main_js() {
    wasm.main_js();
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

let cachedUint32ArrayMemory0 = null;

function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}
/**
 * @returns {number}
 */
export function get_total_squares() {
    const ret = wasm.get_total_squares();
    return ret >>> 0;
}

/**
 * @returns {number}
 */
export function get_board_size() {
    const ret = wasm.get_board_size();
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
export function get_col(sq) {
    const ret = wasm.get_col(sq);
    return ret >>> 0;
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

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}
/**
 * @enum {0 | 1 | 2 | 3}
 */
export const Piece = Object.freeze({
    EMPTY: 0, "0": "EMPTY",
    ATTACKER: 1, "1": "ATTACKER",
    DEFENDER: 2, "2": "DEFENDER",
    KING: 3, "3": "KING",
});
/**
 * @enum {0 | 1}
 */
export const Side = Object.freeze({
    ATTACKERS: 0, "0": "ATTACKERS",
    DEFENDERS: 1, "1": "DEFENDERS",
});

const EngineFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_engine_free(ptr >>> 0, 1));

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

const EngineClientFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_engineclient_free(ptr >>> 0, 1));

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
     * @param {number} time
     * @returns {number}
     */
    make_search(time) {
        const ret = wasm.engineclient_make_search(this.__wbg_ptr, time);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get_w2_first() {
        const ret = wasm.engineclient_get_w2_first(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Side}
     */
    side_to_move() {
        const ret = wasm.engineclient_side_to_move(this.__wbg_ptr);
        return ret;
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
     * @param {number} from
     * @param {number} to
     * @returns {boolean}
     */
    is_move_available(from, to) {
        const ret = wasm.engineclient_is_move_available(this.__wbg_ptr, from, to);
        return ret !== 0;
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
     * @param {number} tt_size_mb
     */
    constructor(tt_size_mb) {
        const ret = wasm.engineclient_new(tt_size_mb);
        this.__wbg_ptr = ret >>> 0;
        EngineClientFinalization.register(this, this.__wbg_ptr, this);
        return this;
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
     * @param {string} fen
     */
    set_fen(fen) {
        const ptr0 = passStringToWasm0(fen, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len0 = WASM_VECTOR_LEN;
        wasm.engineclient_set_fen(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) EngineClient.prototype[Symbol.dispose] = EngineClient.prototype.free;

const MoveFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_move_free(ptr >>> 0, 1));

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
    to() {
        const ret = wasm.move_to(this.__wbg_ptr);
        return ret >>> 0;
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
    from() {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.move_from(ptr);
        return ret >>> 0;
    }
    /**
     * @returns {boolean}
     */
    is_null() {
        const ret = wasm.move_is_null(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {number} mv_u32
     * @returns {Move}
     */
    static from_u32(mv_u32) {
        const ret = wasm.move_from_u32(mv_u32);
        return Move.__wrap(ret);
    }
}
if (Symbol.dispose) Move.prototype[Symbol.dispose] = Move.prototype.free;

const SearchIterationResponseFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_searchiterationresponse_free(ptr >>> 0, 1));

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
     * @returns {number}
     */
    get depth() {
        const ret = wasm.__wbg_get_searchiterationresponse_depth(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set depth(arg0) {
        wasm.__wbg_set_searchiterationresponse_depth(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {Move}
     */
    get mv() {
        const ret = wasm.__wbg_get_searchiterationresponse_mv(this.__wbg_ptr);
        return Move.__wrap(ret);
    }
    /**
     * @param {Move} arg0
     */
    set mv(arg0) {
        _assertClass(arg0, Move);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_searchiterationresponse_mv(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {number}
     */
    get score() {
        const ret = wasm.__wbg_get_searchiterationresponse_score(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set score(arg0) {
        wasm.__wbg_set_searchiterationresponse_score(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {bigint}
     */
    get nodes() {
        const ret = wasm.__wbg_get_searchiterationresponse_nodes(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @param {bigint} arg0
     */
    set nodes(arg0) {
        wasm.__wbg_set_searchiterationresponse_nodes(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {bigint}
     */
    get time() {
        const ret = wasm.__wbg_get_searchiterationresponse_time(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @param {bigint} arg0
     */
    set time(arg0) {
        wasm.__wbg_set_searchiterationresponse_time(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {bigint}
     */
    get speed() {
        const ret = wasm.__wbg_get_searchiterationresponse_speed(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @param {bigint} arg0
     */
    set speed(arg0) {
        wasm.__wbg_set_searchiterationresponse_speed(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) SearchIterationResponse.prototype[Symbol.dispose] = SearchIterationResponse.prototype.free;

const SearchResponseFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_searchresponse_free(ptr >>> 0, 1));

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
     * @param {Move} arg0
     */
    set best_move(arg0) {
        _assertClass(arg0, Move);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_searchresponse_best_move(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {number}
     */
    get score() {
        const ret = wasm.__wbg_get_searchresponse_score(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set score(arg0) {
        wasm.__wbg_set_searchresponse_score(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) SearchResponse.prototype[Symbol.dispose] = SearchResponse.prototype.free;

const TimerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_timer_free(ptr >>> 0, 1));

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

const WasmClientFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmclient_free(ptr >>> 0, 1));

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
    print_board() {
        wasm.wasmclient_print_board(this.__wbg_ptr);
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
    /**
     * @param {string} cmd
     */
    run(cmd) {
        const ptr0 = passStringToWasm0(cmd, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmclient_run(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) WasmClient.prototype[Symbol.dispose] = WasmClient.prototype.free;

const EXPECTED_RESPONSE_TYPES = new Set(['basic', 'cors', 'default']);

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);

            } catch (e) {
                const validResponse = module.ok && EXPECTED_RESPONSE_TYPES.has(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
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
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbg___wbindgen_debug_string_df47ffb5e35e6763 = function(arg0, arg1) {
        const ret = debugString(getObject(arg1));
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbg___wbindgen_is_undefined_2d472862bd29a478 = function(arg0) {
        const ret = getObject(arg0) === undefined;
        return ret;
    };
    imports.wbg.__wbg___wbindgen_throw_b855445ff6a94295 = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbg_call_e762c39fa8ea36bf = function() { return handleError(function (arg0, arg1) {
        const ret = getObject(arg0).call(getObject(arg1));
        return addHeapObject(ret);
    }, arguments) };
    imports.wbg.__wbg_dispatchEvent_6eaea6d4f99ff3bf = function() { return handleError(function (arg0, arg1) {
        const ret = getObject(arg0).dispatchEvent(getObject(arg1));
        return ret;
    }, arguments) };
    imports.wbg.__wbg_error_7534b8e9a36f1ab4 = function(arg0, arg1) {
        let deferred0_0;
        let deferred0_1;
        try {
            deferred0_0 = arg0;
            deferred0_1 = arg1;
            console.error(getStringFromWasm0(arg0, arg1));
        } finally {
            wasm.__wbindgen_export4(deferred0_0, deferred0_1, 1);
        }
    };
    imports.wbg.__wbg_instanceof_EventTarget_735fb26ce859cc99 = function(arg0) {
        let result;
        try {
            result = getObject(arg0) instanceof EventTarget;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_new_1acc0b6eea89d040 = function() {
        const ret = new Object();
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_8a6f238a6ece86ea = function() {
        const ret = new Error();
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_no_args_ee98eee5275000a4 = function(arg0, arg1) {
        const ret = new Function(getStringFromWasm0(arg0, arg1));
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_with_event_init_dict_5aac952397f55283 = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = new CustomEvent(getStringFromWasm0(arg0, arg1), getObject(arg2));
        return addHeapObject(ret);
    }, arguments) };
    imports.wbg.__wbg_now_793306c526e2e3b6 = function() {
        const ret = Date.now();
        return ret;
    };
    imports.wbg.__wbg_set_detail_0fdb00af15229efb = function(arg0, arg1) {
        getObject(arg0).detail = getObject(arg1);
    };
    imports.wbg.__wbg_stack_0ed75d68575b0f3c = function(arg0, arg1) {
        const ret = getObject(arg1).stack;
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbg_static_accessor_GLOBAL_89e1d9ac6a1b250e = function() {
        const ret = typeof global === 'undefined' ? null : global;
        return isLikeNone(ret) ? 0 : addHeapObject(ret);
    };
    imports.wbg.__wbg_static_accessor_GLOBAL_THIS_8b530f326a9e48ac = function() {
        const ret = typeof globalThis === 'undefined' ? null : globalThis;
        return isLikeNone(ret) ? 0 : addHeapObject(ret);
    };
    imports.wbg.__wbg_static_accessor_SELF_6fdf4b64710cc91b = function() {
        const ret = typeof self === 'undefined' ? null : self;
        return isLikeNone(ret) ? 0 : addHeapObject(ret);
    };
    imports.wbg.__wbg_static_accessor_WINDOW_b45bfc5a37f6cfa2 = function() {
        const ret = typeof window === 'undefined' ? null : window;
        return isLikeNone(ret) ? 0 : addHeapObject(ret);
    };
    imports.wbg.__wbindgen_cast_2241b6af4c4b2941 = function(arg0, arg1) {
        // Cast intrinsic for `Ref(String) -> Externref`.
        const ret = getStringFromWasm0(arg0, arg1);
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_cast_d6cd19b81560fd6e = function(arg0) {
        // Cast intrinsic for `F64 -> Externref`.
        const ret = arg0;
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_object_clone_ref = function(arg0) {
        const ret = getObject(arg0);
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_object_drop_ref = function(arg0) {
        takeObject(arg0);
    };

    return imports;
}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedDataViewMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;


    wasm.__wbindgen_start();
    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
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


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('zevratafl_rust_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
