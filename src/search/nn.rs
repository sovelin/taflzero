#[path = "nn_common.rs"]
mod nn_common;
#[cfg(not(target_arch = "wasm32"))]
#[path = "nn_native.rs"]
mod nn_native;
#[cfg(target_arch = "wasm32")]
#[path = "nn_wasm.rs"]
mod nn_wasm;

pub use nn_common::NnOutput;
#[cfg(not(target_arch = "wasm32"))]
pub use nn_native::*;
#[cfg(target_arch = "wasm32")]
pub use nn_wasm::*;
