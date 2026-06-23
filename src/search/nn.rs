#[path = "nn_common.rs"]
mod nn_common;
#[cfg(not(target_arch = "wasm32"))]
#[path = "nn_native.rs"]
mod nn_native;
#[cfg(target_arch = "wasm32")]
#[path = "nn_wasm.rs"]
mod nn_wasm;

pub use nn_common::{NnOutput, POLICY_SIZE, SAMPLE_SIZE, fill_input};
#[cfg(not(target_arch = "wasm32"))]
pub use nn_native::*;
#[cfg(target_arch = "wasm32")]
pub use nn_wasm::*;
