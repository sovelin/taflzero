use std::io::Cursor;
use tract_onnx::prelude::*;
use crate::masks::BOARD_SIZE;
use crate::position_export::BitPosition;
use super::nn_common::{build_input_data, NnOutput, NUM_PLANES, POLICY_SIZE};

const EMBEDDED_MODEL: &[u8] = include_bytes!("../../zero-trainer/weights/gen0051.candidate.fp32.onnx");

pub struct NeuralNet {
    plan: TypedRunnableModel<TypedModel>,
}

impl NeuralNet {
    pub fn new(_path: &str) -> Self {
        println!("[NN] wasm build: using embedded ONNX model (gen0323.onnx)");

        let mut reader = Cursor::new(EMBEDDED_MODEL);
        let model = tract_onnx::onnx()
            .model_for_read(&mut reader)
            .unwrap()
            .into_optimized()
            .unwrap()
            .into_runnable()
            .unwrap();

        Self { plan: model }
    }

    pub fn evaluate_position(&mut self, pos: &BitPosition) -> NnOutput {
        self.evaluate_batch(&[pos]).pop().unwrap()
    }

    pub fn evaluate_batch(&mut self, positions: &[&BitPosition]) -> Vec<NnOutput> {
        let batch_size = positions.len();
        let input_data = build_input_data(positions);

        let input_array = tract_ndarray::ArrayD::from_shape_vec(
            tract_ndarray::IxDyn(&[batch_size, NUM_PLANES, BOARD_SIZE, BOARD_SIZE]),
            input_data,
        )
        .unwrap();

        let input_tensor: Tensor = input_array.into();
        let outputs = self.plan.run(tvec!(input_tensor.into())).unwrap();

        let policy_view = outputs[0].to_array_view::<f32>().unwrap();
        let value_view = outputs[1].to_array_view::<f32>().unwrap();

        let policy_data = policy_view.as_slice().unwrap();
        let value_data = value_view.as_slice().unwrap();

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let mut policy = [0.0f32; POLICY_SIZE];
            policy.copy_from_slice(&policy_data[i * POLICY_SIZE..(i + 1) * POLICY_SIZE]);
            let value = value_data[i];
            results.push(NnOutput { policy, value });
        }

        results
    }
}
