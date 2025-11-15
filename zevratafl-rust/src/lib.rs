use std::panic;
use wasm_bindgen::prelude::wasm_bindgen;

pub mod engine;
pub mod board;
pub mod moves;
pub mod search;
pub mod evaluation;
pub mod utils;
mod tests;
mod timer;
mod uci;
pub mod nnue;
pub mod dataset;

pub use engine::*;
pub use board::*;
pub use moves::*;
pub use search::*;
pub use evaluation::*;
pub use uci::*;

// экспортируем тестовую функцию (для проверки)
#[wasm_bindgen]
pub fn hello() -> String {
    "Hello from Rust!".to_string()
}

#[wasm_bindgen(start)]
pub fn main_js() {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
}

#[wasm_bindgen]
pub fn build_info() -> String {
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };

    profile.to_string()
}