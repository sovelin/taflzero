use ndarray::{Array, IxDyn};
use ort::execution_providers::{CUDAExecutionProvider, DirectMLExecutionProvider};
use ort::session::Session;
use ort::value::Value;
use crate::board::constants::SQS;
use crate::PRECOMPUTED;
use crate::masks::BOARD_SIZE;
use crate::position_export::BitPosition;

const NUM_PLANES: usize = 6;
const POLICY_SIZE: usize = 4840;

pub struct NnOutput {
    pub policy: [f32; POLICY_SIZE],
    pub value: f32,
}

pub struct NeuralNet {
    session: Session,
}

impl NeuralNet {
    pub fn new(path: &str) -> Self {
        let cuda_ep = CUDAExecutionProvider::default().build();
        let dml_ep = DirectMLExecutionProvider::default().build();

        println!("[NN] Loading model: {}", path);
        println!("[NN] Requested EPs: CUDA, DirectML (fallback: CPU)");

        let mut session = Session::builder()
            .unwrap()
            .with_execution_providers([cuda_ep, dml_ep])
            .unwrap()
            .commit_from_file(path)
            .expect("Unable to commit neural net");

        // Warmup + benchmark
        let warmup_input = vec![0.0f32; NUM_PLANES * SQS * 8];
        let warmup_tensor = Array::from_shape_vec(
            IxDyn(&[8, NUM_PLANES, BOARD_SIZE, BOARD_SIZE]),
            warmup_input,
        ).unwrap();
        let warmup_value = Value::from_array(warmup_tensor).unwrap();

        // Warmup run (first call is slow due to kernel compilation)
        let _ = session.run(ort::inputs![warmup_value]).unwrap();

        // Benchmark 10 runs
        let bench_runs = 10;
        let start = std::time::Instant::now();
        for _ in 0..bench_runs {
            let input = vec![0.0f32; NUM_PLANES * SQS * 8];
            let tensor = Array::from_shape_vec(
                IxDyn(&[8, NUM_PLANES, BOARD_SIZE, BOARD_SIZE]),
                input,
            ).unwrap();
            let val = Value::from_array(tensor).unwrap();
            let _ = session.run(ort::inputs![val]).unwrap();
        }
        let elapsed = start.elapsed();
        println!("[NN] Warmup done. batch=8 forward: {:.2}ms avg ({} runs)",
            elapsed.as_secs_f64() / bench_runs as f64 * 1000.0, bench_runs);

        Self { session }
    }

    fn fill_input(input: &mut [f32], pos: &BitPosition) {
        debug_assert!(input.len() == NUM_PLANES * SQS);

        // Planes 0-2: unpack bit planes (attackers, defenders, king)
        for plane in 0..3 {
            let base = plane * 16;
            let out_offset = plane * SQS;
            for idx in 0..SQS {
                let byte = idx / 8;
                let bit = idx % 8;
                if (pos.planes[base + byte] >> bit) & 1 == 1 {
                    input[out_offset + idx] = 1.0;
                }
            }
        }

        // Plane 3: side to move (all 1s if attackers, stm == 1)
        let stm_val = if pos.stm == 1 { 1.0f32 } else { 0.0f32 };
        let stm_offset = 3 * SQS;
        for i in 0..SQS {
            input[stm_offset + i] = stm_val;
        }

        // Plane 4: throne
        let throne_offset = 4 * SQS;
        input[throne_offset + PRECOMPUTED.throne_sq] = 1.0;

        // Plane 5: corners
        let corners_offset = 5 * SQS;
        for &sq in &PRECOMPUTED.corners_sq {
            input[corners_offset + sq] = 1.0;
        }
    }

    pub fn evaluate_position(&mut self, pos: &BitPosition) -> NnOutput {
        self.evaluate_batch(&[pos]).pop().unwrap()
    }

    pub fn evaluate_batch(&mut self, positions: &[&BitPosition]) -> Vec<NnOutput> {
        let batch_size = positions.len();
        let sample_size = NUM_PLANES * SQS;
        let mut input_data = vec![0.0f32; batch_size * sample_size];

        for (i, pos) in positions.iter().enumerate() {
            Self::fill_input(&mut input_data[i * sample_size..(i + 1) * sample_size], pos);
        }

        let input_tensor = Array::from_shape_vec(
            IxDyn(&[batch_size, NUM_PLANES, BOARD_SIZE, BOARD_SIZE]),
            input_data,
        )
        .unwrap();

        let input_value = Value::from_array(input_tensor).unwrap();
        let outputs = self.session.run(ort::inputs![input_value]).unwrap();

        let (_, policy_data) = outputs[0].try_extract_tensor::<f32>().unwrap();
        let (_, value_data) = outputs[1].try_extract_tensor::<f32>().unwrap();

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
