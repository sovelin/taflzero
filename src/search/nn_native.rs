use super::nn_common::{NnInput, NnOutput, PlaneSet, build_input_data};
use crate::board::Precomputed;
use crate::get_policy_size;
use ndarray::{Array, ArrayView, IxDyn};
use ort::environment::Environment;
#[cfg(feature = "cuda")]
use ort::ep::CUDA;
#[cfg(dml)]
use ort::ep::DirectML;
#[cfg(feature = "openvino")]
use ort::ep::OpenVINO;
use ort::session::Session;
use ort::value::{TensorRef, Value, ValueType};
use std::sync::{Arc, OnceLock};

/// The process-wide ORT environment (ort 2.0 rc.14+ builds sessions from one).
fn ort_env() -> Result<&'static Environment, String> {
    static ENV: OnceLock<Environment> = OnceLock::new();
    if let Some(env) = ENV.get() {
        return Ok(env);
    }
    let env = ort::init()
        .with_name("taflzero")
        .build()
        .map_err(|e| format!("cannot initialise ONNX Runtime: {e}"))?;
    Ok(ENV.get_or_init(|| env))
}

use super::nn_batch::BatchServer;

/// Where a net's inferences run: this process's own ORT session, or the shared
/// batching server of a multi-threaded datagen process.
enum Backend {
    Local(Session),
    Shared(Arc<BatchServer>),
}

pub struct NeuralNet {
    backend: Backend,
    /// Where the model was loaded from (empty for `from_bytes`), and the file's
    /// modification time at load, so a worker can pick up a retrained net.
    path: String,
    loaded_mtime: Option<std::time::SystemTime>,
    board_size: usize,
    /// Input planes the net reads (ONNX metadata `planes`; 11 channels without it).
    planes: PlaneSet,
    /// Every call is padded to this many samples. ORT plans execution per input
    /// shape, and a shape it has not seen pays that cost on every run, so the search
    /// must always hand it the one shape the session was warmed up on.
    batch_size: usize,
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

fn file_mtime(path: &str) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

/// Input channel count of the model, when its shape fixes it.
fn model_channels(session: &Session) -> Option<usize> {
    let input = session.inputs().first()?;
    let ValueType::Tensor { shape, .. } = input.dtype() else {
        return None;
    };
    let c = *shape.get(1)?;
    (c >= 1).then_some(c as usize)
}

/// The plane set a net was trained with: ONNX metadata `planes` (written by
/// export_onnx.py), or, for nets exported before it existed, `base` when the input
/// has the 11 channels every such net has. The channel count must match either way.
pub(super) fn model_planes(session: &Session, what: &str) -> Result<PlaneSet, String> {
    let channels = model_channels(session);
    let declared = session
        .metadata()
        .map_err(|e| format!("{what}: cannot read model metadata: {e}"))?
        .custom("planes");
    let planes = match declared {
        Some(spec) => PlaneSet::parse(&spec)
            .map_err(|e| format!("{what}: bad `planes` metadata {spec:?}: {e}"))?,
        None => match channels {
            Some(11) | None => PlaneSet::base(),
            Some(c) => {
                return Err(format!(
                    "{what}: the net reads {c} input planes but has no `planes` metadata; \
                     only 11-plane nets may omit it"
                ));
            }
        },
    };
    if let Some(c) = channels
        && c != planes.len()
    {
        return Err(format!(
            "{what}: `planes` {planes} is {} channels but the net's input has {c}",
            planes.len()
        ));
    }
    Ok(planes)
}

impl NeuralNet {
    /// Loads a net and records the board size it was built for. `board_size` is only a
    /// fallback for models that leave their spatial dimensions dynamic — the model's own
    /// shape wins. Whether that size fits the current board is checked at search time.
    pub fn new(path: &str, board_size: usize, batch_size: usize) -> Result<Self, String> {
        println!("[NN] Loading model: {}", path);
        let (session, board_size, planes) = Self::build_session(path, board_size, batch_size)?;
        Ok(Self {
            backend: Backend::Local(session),
            path: path.to_string(),
            loaded_mtime: file_mtime(path),
            board_size,
            planes,
            batch_size,
        })
    }

    /// A handle onto a shared [`BatchServer`]: every `evaluate_batch` is pooled with
    /// the other threads' requests into one large inference. Reloading is the
    /// server's job, so `reload_if_changed` is a no-op on this handle.
    pub fn shared(server: Arc<BatchServer>) -> Self {
        Self {
            board_size: server.board_size(),
            planes: server.planes().clone(),
            backend: Backend::Shared(server),
            path: String::new(),
            loaded_mtime: None,
            batch_size: 0,
        }
    }

    /// Path the net was loaded from.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Reloads the net when the file at `path` has a newer mtime than the one loaded.
    /// Returns `Ok(true)` after a reload, `Ok(false)` when nothing changed. A file
    /// that cannot be loaded (half-written, corrupt) leaves the current session in
    /// place and reports the error; the next call tries again.
    pub fn reload_if_changed(&mut self) -> Result<bool, String> {
        if self.path.is_empty() || matches!(self.backend, Backend::Shared(_)) {
            return Ok(false);
        }
        let mtime = file_mtime(&self.path);
        if mtime.is_none() || mtime == self.loaded_mtime {
            return Ok(false);
        }
        let (session, board_size, planes) =
            Self::build_session(&self.path, self.board_size, self.batch_size)?;
        if board_size != self.board_size {
            return Err(format!(
                "reloaded net is for a {board_size}x{board_size} board, expected {0}x{0}",
                self.board_size
            ));
        }
        if planes != self.planes {
            return Err(format!(
                "reloaded net reads planes {planes}, the loaded one {}",
                self.planes
            ));
        }
        self.backend = Backend::Local(session);
        self.loaded_mtime = mtime;
        Ok(true)
    }

    /// Builds and warms up a session; returns it with the board size and the plane set
    /// it is built for.
    pub(super) fn build_session(
        path: &str,
        board_size: usize,
        batch_size: usize,
    ) -> Result<(Session, usize, PlaneSet), String> {
        #[cfg(all(feature = "cuda", not(dml)))]
        println!("[NN] Backend: CUDA");
        #[cfg(all(dml, not(feature = "cuda")))]
        println!("[NN] Backend: DirectML");
        #[cfg(feature = "openvino")]
        println!("[NN] Backend: OpenVINO");
        #[cfg(not(any(feature = "cuda", dml, feature = "openvino")))]
        println!("[NN] Backend: CPU");

        let mut builder = Session::builder(ort_env()?).unwrap();

        // Parallelism in datagen comes from running many engine processes, so let each
        // session use a single intra-op thread; the ORT default is one per core, which
        // oversubscribes the machine badly once several workers run at once.
        let intra_threads = std::env::var("TAFLZERO_ORT_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1);
        builder = builder.with_intra_threads(intra_threads).unwrap();

        #[cfg(all(feature = "cuda", dml))]
        {
            // ORT refuses CUDA and DirectML in one session ("DML EP can only be used
            // with CPU EPs"), and a CUDA provider that fails to load is skipped
            // silently, so registering both always ended on DirectML. Try CUDA alone
            // first; fall back to DirectML only when it cannot load.
            match builder.with_execution_providers([CUDA::default().build().error_on_failure()]) {
                Ok(b) => {
                    println!("[NN] Backend: CUDA");
                    builder = b;
                }
                Err(err) => {
                    println!("[NN] Backend: DirectML (CUDA unavailable: {err})");
                    builder = Session::builder(ort_env()?)
                        .unwrap()
                        .with_intra_threads(intra_threads)
                        .unwrap()
                        .with_execution_providers([DirectML::default().build()])
                        .unwrap();
                }
            }
        }
        #[cfg(all(feature = "cuda", not(dml)))]
        {
            builder = builder
                .with_execution_providers([CUDA::default().build()])
                .unwrap();
        }
        #[cfg(all(dml, not(feature = "cuda")))]
        {
            builder = builder
                .with_execution_providers([DirectML::default().build()])
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
        let planes = model_planes(&session, &format!("neural net '{path}'"))?;
        if !planes.is_base() {
            println!("[NN] Planes: {planes} ({} channels)", planes.len());
        }
        let channels = planes.len();

        // Warmup + benchmark
        let warmup_input = vec![0.0f32; planes.sample_size(board_size) * batch_size];
        let warmup_tensor = Array::from_shape_vec(
            IxDyn(&[batch_size, channels, board_size, board_size]),
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
            let input = vec![0.0f32; planes.sample_size(board_size) * batch_size];
            let tensor = Array::from_shape_vec(
                IxDyn(&[batch_size, channels, board_size, board_size]),
                input,
            )
            .unwrap();
            let val = Value::from_array(tensor).unwrap();
            let _ = session.run(ort::inputs![val]).unwrap();
        }
        let elapsed = start.elapsed();
        println!(
            "[NN] Warmup done. batch={} forward: {:.2}ms avg ({} runs)",
            batch_size,
            elapsed.as_secs_f64() / bench_runs as f64 * 1000.0,
            bench_runs
        );

        Ok((session, board_size, planes))
    }

    pub fn from_bytes(data: &[u8], board_size: usize, batch_size: usize) -> Result<Self, String> {
        let session = Session::builder(ort_env()?)
            .map_err(|err| format!("cannot create ONNX session: {err}"))?
            .commit_from_memory(data)
            .map_err(|err| format!("cannot load neural net from bytes: {err}"))?;

        let board_size = model_board_size(&session).unwrap_or(board_size);
        let planes = model_planes(&session, "neural net from bytes")?;

        Ok(Self {
            backend: Backend::Local(session),
            path: String::new(),
            loaded_mtime: None,
            board_size,
            planes,
            batch_size,
        })
    }

    /// Board side this net can evaluate.
    pub fn board_size(&self) -> usize {
        self.board_size
    }

    /// Input planes this net reads; build its inputs with `NnInput::from_board`.
    pub fn planes(&self) -> &PlaneSet {
        &self.planes
    }

    pub fn evaluate_position(&mut self, pos: &NnInput, geom: &Precomputed) -> NnOutput {
        self.evaluate_batch(&[pos], geom).pop().unwrap()
    }

    pub fn evaluate_batch(&mut self, positions: &[&NnInput], geom: &Precomputed) -> Vec<NnOutput> {
        let n = positions.len();
        let input_data = build_input_data(positions, &self.planes, geom);
        match &mut self.backend {
            // Pad up to the warmed-up shape; the tail samples are zeros and discarded.
            Backend::Local(session) => {
                let padded = self.batch_size.max(n);
                let channels = self.planes.len();
                run_inference(session, input_data, n, padded, channels, geom.board_size)
            }
            Backend::Shared(server) => server.evaluate(input_data, n),
        }
    }
}

/// One forward pass: `input_data` holds `n` samples and is padded with zeros to
/// `padded` samples (ORT plans execution per input shape, so callers keep to a few
/// shapes). Returns `n` outputs.
/// One net call's outputs for `n` samples, still flat: `policy` is `n * policy_size`
/// logits, `value` and `moves_left` one number per sample. Reused across calls
/// (`run_inference_flat` takes one back) so the batch server never allocates them anew.
#[derive(Default)]
pub(super) struct FlatOutputs {
    pub policy: Vec<f32>,
    pub value: Vec<f32>,
    pub moves_left: Option<Vec<f32>>,
    pub policy_size: usize,
}

impl FlatOutputs {
    /// Samples `from..to` as per-sample outputs.
    pub fn slice(&self, from: usize, to: usize) -> Vec<NnOutput> {
        (from..to)
            .map(|i| NnOutput {
                policy: self.policy[i * self.policy_size..(i + 1) * self.policy_size].to_vec(),
                value: self.value[i],
                moves_left: self.moves_left.as_ref().map(|d| d[i]),
            })
            .collect()
    }
}

/// Runs the net on the first `padded * sample_size` floats of `input` (resized up with
/// zeros if shorter), of which the first `n` rows are real. The input is only borrowed
/// and `out`'s vectors are refilled in place, so a caller that keeps both allocates
/// nothing per call.
pub(super) fn run_inference_flat(
    session: &mut Session,
    input: &mut Vec<f32>,
    n: usize,
    padded: usize,
    channels: usize,
    board_size: usize,
    mut out: FlatOutputs,
) -> FlatOutputs {
    let sample_size = channels * board_size * board_size;
    input.resize(padded * sample_size, 0.0);

    let view = ArrayView::from_shape(
        IxDyn(&[padded, channels, board_size, board_size]),
        &input[..],
    )
    .unwrap();
    let input_value = TensorRef::from_array_view(view).unwrap();
    let outputs = session.run(ort::inputs![input_value]).unwrap();

    let (_, policy_data) = outputs[0].try_extract_tensor::<f32>().unwrap();
    let (_, value_data) = outputs[1].try_extract_tensor::<f32>().unwrap();

    let policy_size = get_policy_size(board_size);
    out.policy.clear();
    out.policy
        .extend_from_slice(&policy_data[..n * policy_size]);
    out.value.clear();
    out.value.extend_from_slice(&value_data[..n]);
    // Optional third output: a moves-left head. Nets exported without it
    // yield only 2 outputs, so read it only when present (backward compatible
    // with every historical net).
    out.moves_left = if outputs.len() > 2 {
        outputs[2].try_extract_tensor::<f32>().ok().map(|(_, d)| {
            let mut ml = out.moves_left.take().unwrap_or_default();
            ml.clear();
            ml.extend_from_slice(&d[..n]);
            ml
        })
    } else {
        None
    };
    out.policy_size = policy_size;
    out
}

pub(super) fn run_inference(
    session: &mut Session,
    mut input_data: Vec<f32>,
    n: usize,
    padded: usize,
    channels: usize,
    board_size: usize,
) -> Vec<NnOutput> {
    run_inference_flat(
        session,
        &mut input_data,
        n,
        padded,
        channels,
        board_size,
        FlatOutputs::default(),
    )
    .slice(0, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A worker reloads the net when the file changes under it, keeps the old session
    // when the new file is unreadable, and does nothing when nothing changed.
    #[test]
    fn reload_follows_file_mtime_and_survives_garbage() {
        let dir = std::env::temp_dir().join(format!("taflzero-reload-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let copy = dir.join("net.onnx");
        let original = std::fs::read("copenhagen11x11.onnx").unwrap();
        std::fs::write(&copy, &original).unwrap();

        let mut nn = NeuralNet::new(copy.to_str().unwrap(), 11, 8).unwrap();
        assert_eq!(nn.reload_if_changed(), Ok(false));

        // Same bytes, newer mtime: reload.
        let f = std::fs::OpenOptions::new().write(true).open(&copy).unwrap();
        f.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(5))
            .unwrap();
        drop(f);
        assert_eq!(nn.reload_if_changed(), Ok(true));
        assert_eq!(nn.reload_if_changed(), Ok(false));

        // Garbage with a newer mtime: error, old session still answers.
        std::fs::write(&copy, b"not an onnx file").unwrap();
        let f = std::fs::OpenOptions::new().write(true).open(&copy).unwrap();
        f.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(10))
            .unwrap();
        drop(f);
        assert!(nn.reload_if_changed().is_err());
        let mut board = crate::board::Board::new();
        board.set_rules(crate::board::rules::RulesEnum::Copenhagen11x11);
        board.setup_initial_position().unwrap();
        let pos = NnInput::base(crate::board::position_export::BitPosition::from_board(
            &board, 1,
        ));
        let out = nn.evaluate_position(&pos, board.precomputed());
        assert!(out.value.is_finite());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The byte reader of the browser build gives the plane set the session gives, on the
    /// repo's nets and on any extra nets listed in TAFL_ONNX_CHECK (`;`-separated paths).
    #[test]
    fn onnx_bytes_planes_match_the_session() {
        let mut paths: Vec<String> = vec![
            "copenhagen11x11.onnx".into(),
            "tablut9x9.onnx".into(),
            crate::DEFAULT_NET_PATH.into(),
        ];
        if let Ok(extra) = std::env::var("TAFL_ONNX_CHECK") {
            paths.extend(extra.split(';').filter(|p| !p.is_empty()).map(String::from));
        }
        for path in paths {
            let bytes = std::fs::read(&path).unwrap();
            if bytes.starts_with(b"version https://git-lfs") {
                continue;
            }
            let nn = NeuralNet::from_bytes(&bytes, 11, 1).unwrap();
            let from_bytes = crate::search::nn::onnx_planes(&bytes, &path).unwrap();
            assert_eq!(&from_bytes, nn.planes(), "{path}");
            println!("{path}: {} ({} planes)", from_bytes, from_bytes.len());
        }
    }
}
