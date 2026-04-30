use rand::rngs::StdRng;
use rand::SeedableRng;
use crate::capture_gen::CaptureGen;
use crate::movegen::MAX_MOVES;
use crate::moves::movegen::MoveGen;
use crate::moves::mv::Move;
use crate::moves::undo::UndoMove;
use crate::search::constants::MAX_PLY;
use crate::search::history::History;
use crate::search::killer::Killer;
use crate::timer::Timer;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

#[cfg(target_arch = "wasm32")]
use {
    std::cell::RefCell,
    js_sys::Int32Array,
    wasm_bindgen::prelude::*,
};

// Bind Atomics.load from JavaScript for WASM stop-flag polling.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = Atomics, js_name = load, catch)]
    fn atomics_load_i32(typed_array: &JsValue, index: u32) -> Result<i32, JsValue>;
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static WASM_STOP_BUF: RefCell<Option<Int32Array>> = RefCell::new(None);
}

/// Called from WasmClient to register the SharedArrayBuffer-backed Int32Array
/// that the main thread uses to signal stop via `Atomics.store(buf, 0, 1)`.
#[cfg(target_arch = "wasm32")]
pub fn set_wasm_stop_buffer(buf: Int32Array) {
    WASM_STOP_BUF.with(|b| *b.borrow_mut() = Some(buf));
}

#[cfg(target_arch = "wasm32")]
pub fn clear_wasm_stop_buffer() {
    WASM_STOP_BUF.with(|b| *b.borrow_mut() = None);
}

pub struct SearchData {
    pub nodes_searched: u64,
    pub best_move: Option<Move>,
    pub move_gens: Vec<MoveGen>,
    pub capture_gens: Vec<CaptureGen>,
    pub undos: Vec<UndoMove>,
    pub history: History,
    pub killers: Killer,
    pub timer: Timer,
    pub time_limit: u64,
    pub depth_limit: u32,
    cached_exceed: bool,
    time_exceeded_checks: u32,
    pub temperatures: Vec<Vec<i32>>,
    pub temperature: usize,
    pub random_generator: StdRng,
    pub tt_age: u8,
    #[cfg(not(target_arch = "wasm32"))]
    stop_flag: Option<Arc<AtomicBool>>,
}

impl SearchData {
    pub fn new() -> Self {
        let mut move_gens = Vec::with_capacity(MAX_PLY);
        let mut undos = Vec::with_capacity(MAX_PLY);
        let mut capture_gens = Vec::with_capacity(MAX_PLY);

        for _ in 0..MAX_PLY {
            move_gens.push(MoveGen::new());
            capture_gens.push(CaptureGen::new());
            undos.push(UndoMove::new());
        }

        let mut temperatures = Vec::with_capacity(MAX_PLY);
        for _ in 0..MAX_PLY {
            temperatures.push(vec![0; MAX_MOVES]);
        }

        Self {
            nodes_searched: 0,
            best_move: None,
            move_gens,
            capture_gens,
            undos,
            timer: Timer::new(),
            time_limit: 0,
            history: History::new(),
            killers: Killer::new(),
            cached_exceed: false,
            time_exceeded_checks: 0,
            temperatures,
            temperature: 0,
            random_generator: StdRng::seed_from_u64(123456),
            depth_limit: MAX_PLY as u32,
            tt_age: 0,
            #[cfg(not(target_arch = "wasm32"))]
            stop_flag: None,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_stop_flag(&mut self, flag: Arc<AtomicBool>) {
        self.stop_flag = Some(flag);
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn clear_stop_flag(&mut self) {
        self.stop_flag = None;
    }

    pub fn is_stopped(&self) -> bool {
        #[cfg(target_arch = "wasm32")]
        return WASM_STOP_BUF.with(|b| {
            b.borrow().as_ref().map_or(false, |arr| {
                atomics_load_i32(arr.as_ref(), 0).ok().map_or(false, |v| v != 0)
            })
        });

        #[cfg(not(target_arch = "wasm32"))]
        return self.stop_flag.as_ref().map_or(false, |f| f.load(Ordering::Relaxed));
    }

    pub fn time_exceeded(&mut self) -> bool {
        self.timer.elapsed_ms() >= self.time_limit
    }

    pub fn time_exceeded_quick(&mut self) -> bool {
        self.time_exceeded_checks += 1;
        if self.time_exceeded_checks < 10000 {
            return self.cached_exceed
        }

        self.time_exceeded_checks = 0;
        self.cached_exceed = self.timer.elapsed_ms() >= self.time_limit;
        self.cached_exceed
    }

    pub fn start_timer(&mut self, time_limit_ms: u64, depth: u32) {
        self.timer.start();
        self.time_limit = time_limit_ms;
        self.time_exceeded_checks = 0;
        self.cached_exceed = false;
        self.depth_limit = depth;
    }
}