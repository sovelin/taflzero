use crate::timer::Timer;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[cfg(target_arch = "wasm32")]
use {js_sys::Int32Array, std::cell::RefCell, wasm_bindgen::prelude::*};

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
    pub timer: Timer,
    pub time_limit: u64,
    #[cfg(not(target_arch = "wasm32"))]
    stop_flag: Option<Arc<AtomicBool>>,
}

impl Default for SearchData {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchData {
    pub fn new() -> Self {
        Self {
            timer: Timer::new(),
            time_limit: 0,
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
                atomics_load_i32(arr.as_ref(), 0)
                    .ok()
                    .map_or(false, |v| v != 0)
            })
        });

        #[cfg(not(target_arch = "wasm32"))]
        return self
            .stop_flag
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Relaxed));
    }

    pub fn time_exceeded(&mut self) -> bool {
        self.timer.elapsed_ms() >= self.time_limit
    }

    pub fn start_timer(&mut self, time_limit_ms: u64) {
        self.timer.start();
        self.time_limit = time_limit_ms;
    }
}
