//! WASM NN backend.
//!
//! Inference is NOT done in Rust/tract anymore (tract can't handle the SE /
//! global-pool ops of the current architecture).  Instead the browser runs the
//! ONNX model with onnxruntime-web in a dedicated JS worker, and this module
//! talks to it over SharedArrayBuffers:
//!
//!   * control  Int32Array  [REQ, RESP, BATCH]   — Atomics signaling
//!   * input    Float32Array  max_batch * SAMPLE_SIZE
//!   * output   Float32Array  max_batch * (POLICY_SIZE + 1)   (policy then value)
//!
//! The MCTS search stays fully synchronous: `evaluate_batch` writes the input,
//! bumps REQ + notifies, then `Atomics.wait`s (this runs in a Web Worker, so
//! blocking is allowed) until the NN worker sets RESP and notifies back.

use super::nn_common::{NnOutput, POLICY_SIZE, build_input_data};
use crate::board::position_export::BitPosition;
use js_sys::{Atomics, Float32Array, Int32Array};
use std::cell::RefCell;

// control Int32Array indices
const CTRL_REQ: u32 = 0; // engine bumps to request an evaluation
const CTRL_RESP: u32 = 1; // NN worker sets = REQ once results are written
const CTRL_BATCH: u32 = 2; // engine writes the current batch size

thread_local! {
    static NN_CONTROL: RefCell<Option<Int32Array>> = const { RefCell::new(None) };
    static NN_INPUT: RefCell<Option<Float32Array>> = const { RefCell::new(None) };
    static NN_OUTPUT: RefCell<Option<Float32Array>> = const { RefCell::new(None) };
}

/// Register the SharedArrayBuffer-backed views used to talk to the JS NN worker.
/// Call once from JS after constructing the engine, before any search.
///   control: Int32Array, len >= 3
///   input:   Float32Array, len >= max_batch * SAMPLE_SIZE
///   output:  Float32Array, len >= max_batch * (POLICY_SIZE + 1)
pub fn set_nn_buffers(control: Int32Array, input: Float32Array, output: Float32Array) {
    NN_CONTROL.with(|c| *c.borrow_mut() = Some(control));
    NN_INPUT.with(|c| *c.borrow_mut() = Some(input));
    NN_OUTPUT.with(|c| *c.borrow_mut() = Some(output));
}

pub struct NeuralNet;

impl NeuralNet {
    /// No model is loaded in Rust anymore — the JS NN worker owns it.
    pub fn new(_path: &str) -> Self {
        NeuralNet
    }

    pub fn from_bytes(_data: &[u8]) -> Self {
        NeuralNet
    }

    pub fn evaluate_position(&mut self, pos: &BitPosition) -> NnOutput {
        self.evaluate_batch(&[pos]).pop().unwrap()
    }

    pub fn evaluate_batch(&mut self, positions: &[&BitPosition]) -> Vec<NnOutput> {
        let batch = positions.len();
        let input = build_input_data(positions); // batch * SAMPLE_SIZE, NCHW

        // 1. write input into the shared input buffer
        NN_INPUT.with(|b| {
            let b = b.borrow();
            let arr = b
                .as_ref()
                .expect("NN input buffer not registered (set_nn_buffers)");
            arr.subarray(0, input.len() as u32).copy_from(&input);
        });

        // 2. request an evaluation and block until the NN worker answers
        NN_CONTROL.with(|c| {
            let c = c.borrow();
            let ctrl = c
                .as_ref()
                .expect("NN control buffer not registered (set_nn_buffers)");
            let js = ctrl.as_ref(); // &JsValue for js_sys::Atomics
            Atomics::store(js, CTRL_BATCH, batch as i32).unwrap();
            // only the engine writes REQ, so plain load+store is race-free
            let req = Atomics::load(js, CTRL_REQ).unwrap() + 1;
            Atomics::store(js, CTRL_REQ, req).unwrap();
            Atomics::notify(js, CTRL_REQ).unwrap();
            loop {
                let cur = Atomics::load(js, CTRL_RESP).unwrap();
                if cur >= req {
                    break;
                }
                // blocks while control[RESP] == cur; NN worker bumps + notifies
                let _ = Atomics::wait(js, CTRL_RESP, cur);
            }
        });

        // 3. read policy (batch*POLICY_SIZE) then value (batch) from the output buffer
        let mut results = Vec::with_capacity(batch);
        NN_OUTPUT.with(|b| {
            let b = b.borrow();
            let arr = b
                .as_ref()
                .expect("NN output buffer not registered (set_nn_buffers)");
            let policy_len = batch * POLICY_SIZE;
            let mut policy_all = vec![0.0f32; policy_len];
            arr.subarray(0, policy_len as u32).copy_to(&mut policy_all);
            let mut value_all = vec![0.0f32; batch];
            arr.subarray(policy_len as u32, (policy_len + batch) as u32)
                .copy_to(&mut value_all);
            for i in 0..batch {
                let mut policy = [0.0f32; POLICY_SIZE];
                policy.copy_from_slice(&policy_all[i * POLICY_SIZE..(i + 1) * POLICY_SIZE]);
                results.push(NnOutput {
                    policy,
                    value: value_all[i],
                });
            }
        });
        results
    }
}
