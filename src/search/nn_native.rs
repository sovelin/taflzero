use super::nn_common::{NUM_PLANES, NnOutput, build_input_data};
use crate::board::Precomputed;
use crate::board::position_export::BitPosition;
use crate::{get_policy_size, get_sample_size};
use ndarray::{Array, IxDyn};
#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;
#[cfg(feature = "directml")]
use ort::execution_providers::DirectMLExecutionProvider;
#[cfg(feature = "openvino")]
use ort::execution_providers::OpenVINO;
use ort::session::Session;
use ort::value::Value;

pub struct NeuralNet {
    session: Session,
}

impl NeuralNet {
    pub fn new(path: &str, board_size: usize) -> Self {
        println!("[NN] Loading model: {}", path);

        #[cfg(all(feature = "cuda", feature = "directml"))]
        println!("[NN] Backend: CUDA → DirectML → CPU (auto)");
        #[cfg(all(feature = "cuda", not(feature = "directml")))]
        println!("[NN] Backend: CUDA");
        #[cfg(all(feature = "directml", not(feature = "cuda")))]
        println!("[NN] Backend: DirectML");
        #[cfg(feature = "openvino")]
        println!("[NN] Backend: OpenVINO");
        #[cfg(not(any(feature = "cuda", feature = "directml", feature = "openvino")))]
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
        #[cfg(feature = "openvino")]
        {
            builder = builder
                .with_execution_providers([OpenVINO::default().build()])
                .unwrap();
        }

        let mut session = builder
            .commit_from_file(path)
            .expect("Unable to commit neural net");

        // Warmup + benchmark
        let warmup_input = vec![0.0f32; get_sample_size(board_size) * 8];
        let warmup_tensor = Array::from_shape_vec(
            IxDyn(&[8, NUM_PLANES, board_size, board_size]),
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
            let input = vec![0.0f32; get_sample_size(board_size) * 8];
            let tensor =
                Array::from_shape_vec(IxDyn(&[8, NUM_PLANES, board_size, board_size]), input)
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

    pub fn from_bytes(data: &[u8]) -> Self {
        let session = Session::builder()
            .unwrap()
            .commit_from_memory(data)
            .expect("Unable to load neural net from bytes");
        Self { session }
    }

    pub fn evaluate_position(&mut self, pos: &BitPosition, geom: &Precomputed) -> NnOutput {
        self.evaluate_batch(&[pos], geom).pop().unwrap()
    }

    pub fn evaluate_batch(
        &mut self,
        positions: &[&BitPosition],
        geom: &Precomputed,
    ) -> Vec<NnOutput> {
        let batch_size = positions.len();
        let input_data = build_input_data(positions, geom);

        let input_tensor = Array::from_shape_vec(
            IxDyn(&[batch_size, NUM_PLANES, geom.board_size, geom.board_size]),
            input_data,
        )
        .unwrap();

        let input_value = Value::from_array(input_tensor).unwrap();
        let outputs = self.session.run(ort::inputs![input_value]).unwrap();

        let (_, policy_data) = outputs[0].try_extract_tensor::<f32>().unwrap();
        let (_, value_data) = outputs[1].try_extract_tensor::<f32>().unwrap();

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let policy_size = get_policy_size(geom.board_size);
            let policy = policy_data[i * policy_size..(i + 1) * policy_size].to_vec();
            let value = value_data[i];
            results.push(NnOutput { policy, value });
        }

        results
    }
}
