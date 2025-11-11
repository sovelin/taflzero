use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Timer {
    #[cfg(target_arch = "wasm32")]
    start: f64,

    #[cfg(not(target_arch = "wasm32"))]
    time: std::time::Instant,
}

#[wasm_bindgen]
impl Timer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Timer {
        #[cfg(target_arch = "wasm32")]
        {
            Timer { start: js_sys::Date::now() }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Timer { time: std::time::Instant::now() }
        }
    }

    pub fn start(&mut self) {
        #[cfg(target_arch = "wasm32")]
        {
            self.start = js_sys::Date::now();
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.time = std::time::Instant::now();
        }
    }

    pub fn elapsed_ms(&self) -> u64 {
        #[cfg(target_arch = "wasm32")]
        {
            (js_sys::Date::now() - self.start) as u64
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.time.elapsed().as_millis() as u64
        }
    }
}
