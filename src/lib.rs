use std::panic;
use wasm_bindgen::prelude::wasm_bindgen;

pub mod board;
pub mod engine;
pub mod evaluation;
pub mod moves;
mod rules_tests;
pub mod search;
mod tests;
mod timer;
mod uci;
pub mod utils;

pub use engine::*;
pub use evaluation::*;
pub use moves::*;
pub use search::*;
pub use uci::*;

/// `taflzero <cargo version> <git revision>`: what `--version` prints and UCI `id name`
/// carries, so a run can record which engine produced its data.
pub fn engine_version() -> String {
    format!(
        "{} {} {}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        env!("TAFLZERO_GIT")
    )
}

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
