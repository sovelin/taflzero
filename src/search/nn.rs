#[cfg(not(target_arch = "wasm32"))]
#[path = "nn_batch.rs"]
mod nn_batch;
#[path = "nn_common.rs"]
mod nn_common;
#[cfg(not(target_arch = "wasm32"))]
#[path = "nn_native.rs"]
mod nn_native;
#[cfg(target_arch = "wasm32")]
#[path = "nn_wasm.rs"]
mod nn_wasm;

#[cfg(not(target_arch = "wasm32"))]
pub use nn_batch::BatchServer;
pub use nn_common::{
    BASE_PLANES, Extra, NnInput, NnOutput, PlaneSet, build_input_data, fill_input,
    onnx_input_channels, onnx_metadata, onnx_planes,
};
#[cfg(not(target_arch = "wasm32"))]
pub use nn_native::*;
#[cfg(target_arch = "wasm32")]
pub use nn_wasm::*;
