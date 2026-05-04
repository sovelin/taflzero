use std::panic;
use wasm_bindgen::prelude::wasm_bindgen;

pub mod board;
pub mod engine;
pub mod evaluation;
pub mod moves;
pub mod nnue;
pub mod search;
mod tests;
mod timer;
mod uci;
pub mod utils;

pub use board::*;
pub use engine::*;
pub use evaluation::*;
pub use moves::*;
pub use search::*;
pub use uci::*;

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
