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
use ort::value::{Value, ValueType};

pub struct NeuralNet {
    session: Session,
    board_size: usize,
}

/// Board side the model was built for, read from its input shape `[_, planes, N, N]`.
/// `None` when the model leaves the spatial dimensions dynamic, in which case there is
/// nothing to check against.
fn model_board_size(session: &Session) -> Option<usize> {
    let input = session.inputs().first()?;

    let ValueType::Tensor { shape, .. } = input.dtype() else {
        return None;
    };

    let height = *shape.get(2)?;
    let width = *shape.get(3)?;

    if height < 1 || width < 1 || height != width {
        return None;
    }

    Some(height as usize)
}

impl NeuralNet {
    /// Loads a net and records the board size it was built for. `board_size` is only a
    /// fallback for models that leave their spatial dimensions dynamic — the model's own
    /// shape wins. Whether that size fits the current board is checked at search time.
    pub fn new(path: &str, board_size: usize) -> Result<Self, String> {
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

        // Parallelism in datagen comes from running many engine processes, so let each
        // session use a single intra-op thread; the ORT default is one per core, which
        // oversubscribes the machine badly once several workers run at once.
        let intra_threads = std::env::var("TAFLZERO_ORT_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1);
        builder = builder.with_intra_threads(intra_threads).unwrap();

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
            .map_err(|err| format!("cannot load neural net '{path}': {err}"))?;

        let board_size = model_board_size(&session).unwrap_or(board_size);

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

        Ok(Self {
            session,
            board_size,
        })
    }

    pub fn from_bytes(data: &[u8], board_size: usize) -> Result<Self, String> {
        let session = Session::builder()
            .map_err(|err| format!("cannot create ONNX session: {err}"))?
            .commit_from_memory(data)
            .map_err(|err| format!("cannot load neural net from bytes: {err}"))?;

        let board_size = model_board_size(&session).unwrap_or(board_size);

        Ok(Self {
            session,
            board_size,
        })
    }

    /// Board side this net can evaluate.
    pub fn board_size(&self) -> usize {
        self.board_size
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
