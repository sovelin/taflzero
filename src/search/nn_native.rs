use super::nn_common::{NUM_PLANES, NnOutput, POLICY_SIZE, SAMPLE_SIZE, build_input_data};
use crate::masks::BOARD_SIZE;
use crate::position_export::BitPosition;
use ndarray::{Array, IxDyn};
#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;
#[cfg(feature = "directml")]
use ort::execution_providers::DirectMLExecutionProvider;
use ort::session::Session;
use ort::value::Value;

pub struct NeuralNet {
    session: Session,
}

impl NeuralNet {
    pub fn new(path: &str) -> Self {
        println!("[NN] Loading model: {}", path);

        #[cfg(all(feature = "cuda", feature = "directml"))]
        println!("[NN] Backend: CUDA → DirectML → CPU (auto)");
        #[cfg(all(feature = "cuda", not(feature = "directml")))]
        println!("[NN] Backend: CUDA");
        #[cfg(all(feature = "directml", not(feature = "cuda")))]
        println!("[NN] Backend: DirectML");
        #[cfg(not(any(feature = "cuda", feature = "directml")))]
        println!("[NN] Backend: CPU");

        let mut builder = Session::builder().unwrap();

        #[cfg(all(feature = "cuda", feature = "directml"))]
        {
            builder = builder
                .with_execution_providers([
                    CUDAExecutionProvider::default().build(),
                    DirectMLExecutionProvider::default().build(),
                ])
                .unwrap();
        }
        #[cfg(all(feature = "cuda", not(feature = "directml")))]
        {
            builder = builder
                .with_execution_providers([CUDAExecutionProvider::default().build()])
                .unwrap();
        }
        #[cfg(all(feature = "directml", not(feature = "cuda")))]
        {
            builder = builder
                .with_execution_providers([DirectMLExecutionProvider::default().build()])
                .unwrap();
        }

        let mut session = builder
            .commit_from_file(path)
            .expect("Unable to commit neural net");

        // Warmup + benchmark
        let warmup_input = vec![0.0f32; SAMPLE_SIZE * 8];
        let warmup_tensor = Array::from_shape_vec(
            IxDyn(&[8, NUM_PLANES, BOARD_SIZE, BOARD_SIZE]),
            warmup_input,
        )
        .unwrap();
        let warmup_value = Value::from_array(warmup_tensor).unwrap();

        // Warmup run (first call is slow due to kernel compilation)
        let _ = session.run(ort::inputs![warmup_value]).unwrap();

        // Benchmark 10 runs
        let bench_runs = 10;
        let start = std::time::Instant::now();
        for _ in 0..bench_runs {
            let input = vec![0.0f32; SAMPLE_SIZE * 8];
            let tensor =
                Array::from_shape_vec(IxDyn(&[8, NUM_PLANES, BOARD_SIZE, BOARD_SIZE]), input)
                    .unwrap();
            let val = Value::from_array(tensor).unwrap();
            let _ = session.run(ort::inputs![val]).unwrap();
        }
        let elapsed = start.elapsed();
        println!(
            "[NN] Warmup done. batch=8 forward: {:.2}ms avg ({} runs)",
            elapsed.as_secs_f64() / bench_runs as f64 * 1000.0,
            bench_runs
        );

        Self { session }
    }

    pub fn evaluate_position(&mut self, pos: &BitPosition) -> NnOutput {
        self.evaluate_batch(&[pos]).pop().unwrap()
    }

    pub fn evaluate_batch(&mut self, positions: &[&BitPosition]) -> Vec<NnOutput> {
        let batch_size = positions.len();
        let input_data = build_input_data(positions);

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
