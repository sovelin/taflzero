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

/// The CUDA provider with the optional tuning switches from the environment
/// (`TAFLZERO_CUDA_NHWC`, `TAFLZERO_CUDA_FUSE_BIAS`, `TAFLZERO_CUDA_TF32` = 1); unset,
/// it is ORT's default provider.
#[cfg(feature = "cuda")]
fn cuda_provider() -> CUDA {
    let on = |name: &str| std::env::var(name).is_ok_and(|v| v == "1");
    let mut ep = CUDA::default();
    if on("TAFLZERO_CUDA_NHWC") {
        ep = ep.with_prefer_nhwc(true);
    }
    if on("TAFLZERO_CUDA_FUSE_BIAS") {
        ep = ep.with_fuse_conv_bias(true);
    }
    if on("TAFLZERO_CUDA_TF32") {
        ep = ep.with_tf32(true);
    }
    ep
}

/// Where a net's inferences run: this process's own ORT session, the same with a
/// CUDA graph (fixed batch, see [`GraphSession`]), or the shared batching server of a
/// multi-threaded datagen process.
enum Backend {
    Local(Session),
    #[cfg(feature = "cuda")]
    Graph(Box<GraphSession>),
    Shared(Arc<BatchServer>),
}

/// Whether `NeuralNet::new` builds CUDA-graph sessions: `TAFLZERO_CUDA_GRAPH=1`.
#[cfg(feature = "cuda")]
fn graph_requested() -> bool {
    std::env::var("TAFLZERO_CUDA_GRAPH").is_ok_and(|v| v == "1")
}

/// Whether `build_session` puts TensorRT in front of CUDA: `TAFLZERO_TRT=1`.
#[cfg(feature = "tensorrt")]
fn trt_requested() -> bool {
    std::env::var("TAFLZERO_TRT").is_ok_and(|v| v == "1")
}

/// The TensorRT provider for the net at `path`, called with up to `batch` samples:
/// fp16, one optimisation profile from 1 to `batch` samples, tuned for `batch`. The
/// engine is built in memory at every load (seconds, most of it kernel timing, which
/// the timing cache in `TAFLZERO_TRT_CACHE`, default `trt-cache`, keeps across loads);
/// it is not cached on disk, since every net a training run exports is a new engine.
#[cfg(feature = "tensorrt")]
fn trt_provider(path: &str, board_size: usize, batch: usize) -> Result<ort::ep::TensorRT, String> {
    let probe = Session::builder(ort_env()?)
        .map_err(|e| format!("session builder: {e}"))?
        .commit_from_file(path)
        .map_err(|err| format!("cannot load neural net '{path}': {err}"))?;
    let input = probe.inputs()[0].name().to_string();
    let size = model_board_size(&probe).unwrap_or(board_size);
    let channels = match model_channels(&probe) {
        Some(c) => c,
        None => model_planes(&probe, path)?.len(),
    };
    let shape = |b: usize| format!("{input}:{b}x{channels}x{size}x{size}");
    let cache = std::env::var("TAFLZERO_TRT_CACHE").unwrap_or_else(|_| "trt-cache".into());
    std::fs::create_dir_all(&cache).map_err(|e| format!("{cache}: {e}"))?;
    Ok(ort::ep::TensorRT::default()
        .with_fp16(true)
        .with_timing_cache(true)
        .with_timing_cache_path(&cache)
        .with_profile_min_shapes(shape(1))
        .with_profile_opt_shapes(shape(batch.max(1)))
        .with_profile_max_shapes(shape(batch.max(1))))
}

#[cfg(feature = "cuda")]
fn ort_err<E: std::fmt::Display>(what: &str) -> impl Fn(E) -> String + '_ {
    move |err| format!("{what}: {err}")
}

/// A session that replays a captured CUDA graph. A graph replays fixed device
/// addresses, so the input is one device buffer and every output one device buffer,
/// all bound once at a fixed batch size; a call copies the (zero-padded) input in from
/// a pinned host buffer, replays, and copies the outputs out. Small batches gain the
/// most: one call is a single graph launch instead of one launch per kernel.
#[cfg(feature = "cuda")]
struct GraphInner {
    session: Session,
    binding: ort::session::IoBinding,
    host_in: ort::value::Tensor<f32>,
    dev_in: ort::value::Tensor<f32>,
    host_out: Vec<ort::value::Tensor<f32>>,
    batch: usize,
    sample_size: usize,
    policy_size: usize,
}

/// Binds `value` itself as input `name`. `IoBinding::bind_input` binds a copy of a
/// tensor it may alias (any `Tensor::new` one), so writes into `value` afterwards
/// never reach the session; a graph needs the very buffer it captured. `value` must
/// outlive the binding (the caller keeps both in one struct).
#[cfg(feature = "cuda")]
fn bind_input_in_place(
    binding: &mut ort::session::IoBinding,
    name: &str,
    value: &ort::value::Tensor<f32>,
) -> Result<(), String> {
    use ort::AsPointer;
    let name = std::ffi::CString::new(name).map_err(|e| format!("input name: {e}"))?;
    let api = ort::api();
    // SAFETY: both pointers are live ORT objects; ORT keeps a reference to the value
    // and the caller keeps the value alive as long as the binding.
    unsafe {
        let status = (api.BindInput)(binding.ptr_mut(), name.as_ptr(), value.ptr());
        if !status.0.is_null() {
            let msg = std::ffi::CStr::from_ptr((api.GetErrorMessage)(status.0))
                .to_string_lossy()
                .into_owned();
            (api.ReleaseStatus)(status.0);
            return Err(format!("bind input: {msg}"));
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
impl GraphInner {
    /// Builds the session for `batch` samples per call and runs it a few times (the
    /// first run captures the graph); returns it with the board size and plane set.
    fn new(path: &str, board_size: usize, batch: usize) -> Result<(Self, usize, PlaneSet), String> {
        use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
        use ort::value::Tensor;
        let e = ort_err::<ort::Error>;
        let intra_threads = std::env::var("TAFLZERO_ORT_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1);
        let session = Session::builder(ort_env()?)
            .map_err(ort_err("session builder"))?
            .with_intra_threads(intra_threads)
            .map_err(ort_err("intra threads"))?
            .with_execution_providers([cuda_provider()
                .with_cuda_graph(true)
                .build()
                .error_on_failure()])
            .map_err(ort_err("CUDA graph provider"))?
            .commit_from_file(path)
            .map_err(|err| format!("cannot load neural net '{path}': {err}"))?;
        let board_size = model_board_size(&session).unwrap_or(board_size);
        let planes = model_planes(&session, &format!("neural net '{path}'"))?;
        let channels = planes.len();
        let policy_size = get_policy_size(board_size);
        let device = Allocator::new(
            &session,
            MemoryInfo::new(
                AllocationDevice::CUDA,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            )
            .map_err(e("cuda memory"))?,
        )
        .map_err(e("cuda allocator"))?;
        let pinned = Allocator::new(
            &session,
            MemoryInfo::new(
                AllocationDevice::CUDA_PINNED,
                0,
                AllocatorType::Device,
                MemoryType::CPUOutput,
            )
            .map_err(e("pinned memory"))?,
        )
        .map_err(e("pinned allocator"))?;
        let shape = [batch, channels, board_size, board_size];
        let host_in = Tensor::<f32>::new(&pinned, shape).map_err(e("host input"))?;
        let dev_in = Tensor::<f32>::new(&device, shape).map_err(e("device input"))?;
        let mut binding = session.create_binding().map_err(e("binding"))?;
        bind_input_in_place(&mut binding, session.inputs()[0].name(), &dev_in)?;
        // Output 0 is the policy, the others (value, moves-left) one number per sample.
        let mut host_out = Vec::new();
        for (i, outlet) in session.outputs().iter().enumerate() {
            let out_shape = if i == 0 {
                vec![batch, policy_size]
            } else {
                vec![batch, 1]
            };
            let dev = Tensor::<f32>::new(&device, out_shape.clone()).map_err(e("device output"))?;
            binding
                .bind_output(outlet.name(), dev)
                .map_err(e("bind output"))?;
            // A pinned target trips ORT's copy (`dst_data != src_data`); plain host memory works.
            host_out.push(
                Tensor::<f32>::new(&Allocator::default(), out_shape).map_err(e("host output"))?,
            );
        }
        let mut gs = GraphInner {
            session,
            binding,
            host_in,
            dev_in,
            host_out,
            batch,
            sample_size: planes.sample_size(board_size),
            policy_size,
        };
        let zeros = vec![0.0f32; gs.sample_size];
        let start = std::time::Instant::now();
        for _ in 0..3 {
            gs.run(&zeros, 1)?;
        }
        println!(
            "[NN] CUDA graph session: batch={batch}, warmup {:.1} ms",
            start.elapsed().as_secs_f64() * 1000.0
        );
        Ok((gs, board_size, planes))
    }

    /// Runs the first `n` (<= batch) samples of `input`; returns their outputs, flat.
    fn run(&mut self, input: &[f32], n: usize) -> Result<FlatOutputs, String> {
        let e = ort_err::<ort::Error>;
        assert!(n <= self.batch && input.len() >= n * self.sample_size);
        {
            let (_, data) = self.host_in.extract_tensor_mut();
            let len = n * self.sample_size;
            data[..len].copy_from_slice(&input[..len]);
            data[len..].fill(0.0);
        }
        // Synchronous: the copy runs on another session's stream, and the graph must
        // not start before it has landed.
        self.host_in
            .copy_into(&mut self.dev_in)
            .map_err(e("copy in"))?;
        let outputs = self.session.run_binding(&self.binding).map_err(e("run"))?;
        for (i, h) in self.host_out.iter_mut().enumerate() {
            outputs[i]
                .view()
                .downcast::<ort::value::TensorValueType<f32>>()
                .map_err(e("output type"))?
                .copy_into(&mut *h)
                .map_err(e("copy out"))?;
        }
        let ps = self.policy_size;
        Ok(FlatOutputs {
            policy: self.host_out[0].extract_tensor().1[..n * ps].to_vec(),
            value: self.host_out[1].extract_tensor().1[..n].to_vec(),
            moves_left: self
                .host_out
                .get(2)
                .map(|t| t.extract_tensor().1[..n].to_vec()),
            policy_size: ps,
        })
    }
}

/// A request to the graph thread: `n` samples of input and where to send the answer.
#[cfg(feature = "cuda")]
type GraphJob = (
    Vec<f32>,
    usize,
    std::sync::mpsc::SyncSender<Result<FlatOutputs, String>>,
);

/// A [`GraphInner`] confined to a thread of its own. ORT's CUDA provider keeps
/// per-thread state, and with a CUDA graph that state holds a captured graph and its
/// memory: UCI runs every search on a fresh thread, and each one leaked about 12 MB per
/// move. Here the session is created, run and dropped on one long-lived thread, and
/// callers on any thread hand it their inputs over a channel.
#[cfg(feature = "cuda")]
pub(super) struct GraphSession {
    jobs: Option<std::sync::mpsc::SyncSender<GraphJob>>,
    thread: Option<std::thread::JoinHandle<()>>,
    pub(super) batch: usize,
    pub(super) sample_size: usize,
}

#[cfg(feature = "cuda")]
impl GraphSession {
    /// Starts the thread and builds the session on it; returns it with the board size
    /// and plane set of the net.
    pub(super) fn new(
        path: &str,
        board_size: usize,
        batch: usize,
    ) -> Result<(Self, usize, PlaneSet), String> {
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let (jobs_tx, jobs_rx) = std::sync::mpsc::sync_channel::<GraphJob>(1);
        let path = path.to_string();
        let thread = std::thread::Builder::new()
            .name("nn-graph".into())
            .spawn(move || {
                let mut inner = match GraphInner::new(&path, board_size, batch) {
                    Ok((inner, b, planes)) => {
                        let _ = ready_tx.send(Ok((b, planes, inner.sample_size)));
                        inner
                    }
                    Err(err) => {
                        let _ = ready_tx.send(Err(err));
                        return;
                    }
                };
                for (input, n, reply) in jobs_rx {
                    let _ = reply.send(inner.run(&input, n));
                }
            })
            .map_err(|e| format!("cannot start the CUDA graph thread: {e}"))?;
        let (board_size, planes, sample_size) = ready_rx
            .recv()
            .map_err(|_| "the CUDA graph thread died while loading".to_string())??;
        Ok((
            GraphSession {
                jobs: Some(jobs_tx),
                thread: Some(thread),
                batch,
                sample_size,
            },
            board_size,
            planes,
        ))
    }

    /// Runs the first `n` (<= batch) samples of `input` on the graph thread.
    pub(super) fn run(&mut self, input: &[f32], n: usize) -> Result<FlatOutputs, String> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let job = input[..n * self.sample_size].to_vec();
        self.jobs
            .as_ref()
            .expect("graph thread stopped")
            .send((job, n, tx))
            .map_err(|_| "the CUDA graph thread is gone".to_string())?;
        rx.recv()
            .map_err(|_| "the CUDA graph thread is gone".to_string())?
    }

    /// Any number of samples, in chunks of the graph's batch.
    fn evaluate(&mut self, input: &[f32], n: usize) -> Vec<NnOutput> {
        let mut out = Vec::with_capacity(n);
        let mut done = 0;
        while done < n {
            let k = (n - done).min(self.batch);
            let flat = self
                .run(&input[done * self.sample_size..], k)
                .unwrap_or_else(|err| panic!("CUDA graph inference failed: {err}"));
            out.extend(flat.slice(0, k));
            done += k;
        }
        out
    }
}

#[cfg(feature = "cuda")]
impl Drop for GraphSession {
    /// Closing the channel ends the thread, which drops the session where it was made.
    fn drop(&mut self) {
        drop(self.jobs.take());
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
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
        #[cfg(feature = "cuda")]
        if graph_requested() {
            let (gs, board_size, planes) = GraphSession::new(path, board_size, batch_size.max(1))?;
            return Ok(Self {
                backend: Backend::Graph(Box::new(gs)),
                path: path.to_string(),
                loaded_mtime: file_mtime(path),
                board_size,
                planes,
                batch_size,
            });
        }
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
        #[cfg(feature = "cuda")]
        let graph = matches!(self.backend, Backend::Graph(_));
        #[cfg(not(feature = "cuda"))]
        let graph = false;
        let (backend, board_size, planes) = if graph {
            #[cfg(feature = "cuda")]
            {
                let (gs, b, p) =
                    GraphSession::new(&self.path, self.board_size, self.batch_size.max(1))?;
                (Backend::Graph(Box::new(gs)), b, p)
            }
            #[cfg(not(feature = "cuda"))]
            unreachable!()
        } else {
            let (session, b, p) =
                Self::build_session(&self.path, self.board_size, self.batch_size)?;
            (Backend::Local(session), b, p)
        };
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
        self.backend = backend;
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
        #[cfg(feature = "tensorrt")]
        let trt = trt_requested();
        #[cfg(not(feature = "tensorrt"))]
        let trt = false;
        Self::build_session_on(path, board_size, batch_size, trt)
    }

    /// `build_session`, with TensorRT in front of CUDA when `trt` (tensorrt builds).
    fn build_session_on(
        path: &str,
        board_size: usize,
        batch_size: usize,
        trt: bool,
    ) -> Result<(Session, usize, PlaneSet), String> {
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

        #[cfg(feature = "tensorrt")]
        if trt {
            println!("[NN] Backend: TensorRT fp16 (CUDA for what it does not take)");
            builder = builder
                .with_execution_providers([
                    trt_provider(path, board_size, batch_size)?
                        .build()
                        .error_on_failure(),
                    cuda_provider().build(),
                ])
                .map_err(|e| format!("TensorRT provider: {e}"))?;
        }

        #[cfg(all(feature = "cuda", dml))]
        if !trt {
            // ORT refuses CUDA and DirectML in one session ("DML EP can only be used
            // with CPU EPs"), and a CUDA provider that fails to load is skipped
            // silently, so registering both always ended on DirectML. Try CUDA alone
            // first; fall back to DirectML only when it cannot load.
            match builder.with_execution_providers([cuda_provider().build().error_on_failure()]) {
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
        if !trt {
            // A CUDA provider that fails to load (no CUDA 13 libraries, no GPU) is
            // skipped silently and the session runs on the CPU; say which one it is.
            match builder.with_execution_providers([cuda_provider().build().error_on_failure()]) {
                Ok(b) => {
                    println!("[NN] Backend: CUDA");
                    builder = b;
                }
                Err(err) => {
                    println!("[NN] Backend: CPU (CUDA unavailable: {err})");
                    builder = Session::builder(ort_env()?)
                        .unwrap()
                        .with_intra_threads(intra_threads)
                        .unwrap();
                }
            }
        }
        #[cfg(all(dml, not(feature = "cuda")))]
        if !trt {
            builder = builder
                .with_execution_providers([DirectML::default().build()])
                .unwrap();
        }
        #[cfg(feature = "openvino")]
        if !trt {
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
            #[cfg(feature = "cuda")]
            Backend::Graph(gs) => gs.evaluate(&input_data, n),
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

/// One row of `--bench-nn`: a batch size and the wall time of one net call on it.
#[derive(Debug, Clone)]
pub struct BenchRow {
    pub batch: usize,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub samples_per_s: f64,
}

/// Times net calls the way the engine makes them, for `--bench-nn`: per batch size a
/// fresh session warmed up on that shape, then `iters` calls through the path the
/// batch server uses (`run_inference_flat`, buffers reused). Inputs are a fixed
/// pseudo-random 0/1 pattern, since an all-zero input could take shortcuts.
pub fn bench_nn(
    path: &str,
    board_size: usize,
    batches: &[usize],
    iters: usize,
) -> Result<Vec<BenchRow>, String> {
    let mut rows = Vec::with_capacity(batches.len());
    for &batch in batches {
        let (mut session, board_size, planes) = NeuralNet::build_session(path, board_size, batch)?;
        let channels = planes.len();
        let sample = planes.sample_size(board_size);
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let pattern: Vec<f32> = (0..batch * sample)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state & 7 == 0) as u8 as f32
            })
            .collect();
        let mut input = pattern.clone();
        let mut out = FlatOutputs::default();
        for _ in 0..3 {
            out = run_inference_flat(
                &mut session,
                &mut input,
                batch,
                batch,
                channels,
                board_size,
                out,
            );
        }
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters.max(1) {
            input.clear();
            input.extend_from_slice(&pattern);
            let t = std::time::Instant::now();
            out = run_inference_flat(
                &mut session,
                &mut input,
                batch,
                batch,
                channels,
                board_size,
                out,
            );
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.total_cmp(b));
        let pick = |q: f64| times[((times.len() - 1) as f64 * q).round() as usize];
        let p50 = pick(0.5);
        rows.push(BenchRow {
            batch,
            p50_ms: p50,
            p90_ms: pick(0.9),
            samples_per_s: batch as f64 * 1000.0 / p50,
        });
    }
    Ok(rows)
}

/// `--bench-nn --graph`: the same timing through a [`GraphSession`] (CUDA graph, the
/// input copied in and the outputs copied out on every call).
#[cfg(feature = "cuda")]
pub fn bench_nn_graph(
    path: &str,
    board_size: usize,
    batches: &[usize],
    iters: usize,
) -> Result<Vec<BenchRow>, String> {
    let mut rows = Vec::with_capacity(batches.len());
    for &batch in batches {
        let (mut gs, _, _) = GraphSession::new(path, board_size, batch)?;
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let input: Vec<f32> = (0..batch * gs.sample_size)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state & 7 == 0) as u8 as f32
            })
            .collect();
        for _ in 0..3 {
            gs.run(&input, batch)?;
        }
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters.max(1) {
            let t = std::time::Instant::now();
            std::hint::black_box(gs.run(&input, batch)?);
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.total_cmp(b));
        let pick = |q: f64| times[((times.len() - 1) as f64 * q).round() as usize];
        let p50 = pick(0.5);
        rows.push(BenchRow {
            batch,
            p50_ms: p50,
            p90_ms: pick(0.9),
            samples_per_s: batch as f64 * 1000.0 / p50,
        });
    }
    Ok(rows)
}

/// `--bench-nn --trt`: the plain timing with the TensorRT provider (fp16, CUDA after
/// it for anything TensorRT does not take). Engines are built for one optimisation
/// profile per batch size and cached in `TAFLZERO_TRT_CACHE` (default `trt-cache`);
/// the first build of a profile takes a while and is not timed.
#[cfg(feature = "tensorrt")]
pub fn bench_nn_trt(
    path: &str,
    board_size: usize,
    batches: &[usize],
    iters: usize,
) -> Result<Vec<BenchRow>, String> {
    use ort::ep::TensorRT;
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let planes = crate::search::nn::onnx_planes(&bytes, path)?;
    let channels = planes.len();
    let cache = std::env::var("TAFLZERO_TRT_CACHE").unwrap_or_else(|_| "trt-cache".into());
    std::fs::create_dir_all(&cache).map_err(|e| format!("{cache}: {e}"))?;
    let mut rows = Vec::with_capacity(batches.len());
    for &batch in batches {
        let probe = Session::builder(ort_env()?)
            .map_err(|e| format!("session builder: {e}"))?
            .commit_from_memory(&bytes)
            .map_err(|e| format!("{path}: {e}"))?;
        let input = probe.inputs()[0].name().to_string();
        drop(probe);
        let shape = |b: usize| format!("{input}:{b}x{channels}x{board_size}x{board_size}");
        let trt = TensorRT::default()
            .with_fp16(true)
            .with_engine_cache(true)
            .with_engine_cache_path(&cache)
            .with_timing_cache(true)
            .with_timing_cache_path(&cache)
            .with_profile_min_shapes(shape(batch))
            .with_profile_opt_shapes(shape(batch))
            .with_profile_max_shapes(shape(batch));
        let t0 = std::time::Instant::now();
        let mut session = Session::builder(ort_env()?)
            .map_err(|e| format!("session builder: {e}"))?
            .with_intra_threads(1)
            .map_err(|e| format!("intra threads: {e}"))?
            .with_execution_providers([trt.build().error_on_failure(), cuda_provider().build()])
            .map_err(|e| format!("TensorRT provider: {e}"))?
            .commit_from_memory(&bytes)
            .map_err(|e| format!("{path}: {e}"))?;
        let sample = planes.sample_size(board_size);
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let pattern: Vec<f32> = (0..batch * sample)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state & 7 == 0) as u8 as f32
            })
            .collect();
        let mut input_buf = pattern.clone();
        let mut out = FlatOutputs::default();
        for _ in 0..3 {
            out = run_inference_flat(
                &mut session,
                &mut input_buf,
                batch,
                batch,
                channels,
                board_size,
                out,
            );
        }
        println!(
            "[NN] TensorRT batch {batch}: session + engine {:.1} s",
            t0.elapsed().as_secs_f64()
        );
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters.max(1) {
            input_buf.clear();
            input_buf.extend_from_slice(&pattern);
            let t = std::time::Instant::now();
            out = run_inference_flat(
                &mut session,
                &mut input_buf,
                batch,
                batch,
                channels,
                board_size,
                out,
            );
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.total_cmp(b));
        let pick = |q: f64| times[((times.len() - 1) as f64 * q).round() as usize];
        let p50 = pick(0.5);
        rows.push(BenchRow {
            batch,
            p50_ms: p50,
            p90_ms: pick(0.9),
            samples_per_s: batch as f64 * 1000.0 / p50,
        });
    }
    Ok(rows)
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

    /// The CUDA-graph session answers like the plain one: same policy and value (and
    /// moves-left) on real positions, for a full batch, one sample, and more samples
    /// than the graph's batch. GPU only (`cargo test --release -- --ignored`); the net
    /// is copenhagen11x11.onnx, or `TAFL_GRAPH_NET` (e.g. a 24-plane fp16 T9 net).
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn graph_session_matches_the_plain_session() {
        use crate::board::Board;
        use crate::board::rules::RulesEnum;
        use crate::mcts::get_left_moves;
        use crate::movegen::MoveGen;
        let path =
            std::env::var("TAFL_GRAPH_NET").unwrap_or_else(|_| "copenhagen11x11.onnx".into());
        let batch = 8;
        let mut plain = NeuralNet::new(&path, 11, batch).unwrap();
        let (gs, _, planes) = GraphSession::new(&path, 11, batch).unwrap();
        let mut graph = NeuralNet {
            backend: Backend::Graph(Box::new(gs)),
            path: path.clone(),
            loaded_mtime: None,
            board_size: plain.board_size(),
            planes,
            batch_size: batch,
        };
        // Positions from a few pseudo-random games.
        let mut board = Board::new();
        board.set_rules(RulesEnum::Copenhagen11x11);
        board.setup_initial_position().unwrap();
        let mut mg = MoveGen::new();
        let mut inputs = Vec::new();
        let mut state = 12345u64;
        while inputs.len() < 40 {
            inputs.push(NnInput::from_board(&mut board, &mut mg, plain.planes()));
            let moves = get_left_moves(&board, &mut mg);
            if moves.is_empty() || crate::terminal::check_terminal(&mut board).is_some() {
                board.setup_initial_position().unwrap();
                continue;
            }
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            board
                .make_move_simple(moves[(state >> 33) as usize % moves.len()])
                .unwrap();
        }
        let geom = board.precomputed().clone();
        let mut worst = (0.0f32, 0.0f32);
        for chunk in [&inputs[..8], &inputs[8..9], &inputs[9..22]] {
            let refs: Vec<&NnInput> = chunk.iter().collect();
            let a = plain.evaluate_batch(&refs, &geom);
            let b = graph.evaluate_batch(&refs, &geom);
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(&b) {
                let dp = x
                    .policy
                    .iter()
                    .zip(&y.policy)
                    .map(|(p, q)| (p - q).abs())
                    .fold(0.0, f32::max);
                worst = (worst.0.max(dp), worst.1.max((x.value - y.value).abs()));
                assert_eq!(x.moves_left.is_some(), y.moves_left.is_some());
                if let (Some(m), Some(n)) = (x.moves_left, y.moves_left) {
                    assert!((m - n).abs() < 1e-2, "moves-left {m} vs {n}");
                }
            }
        }
        println!(
            "{path}: max |policy logit diff| {:.2e}, max |value diff| {:.2e}",
            worst.0, worst.1
        );
        // fp16 nets differ by kernel choice; fp32 ones should agree to ~1e-4.
        assert!(worst.0 < 5e-2 && worst.1 < 1e-2, "{worst:?}");
    }

    /// The TensorRT session (fp16) answers like the plain one on real positions, for
    /// a full batch, one sample and a batch in between. GPU only, tensorrt builds
    /// (`cargo test --release --features tensorrt -- --ignored`, nvinfer on PATH); the
    /// net is copenhagen11x11.onnx, or `TAFL_GRAPH_NET`.
    #[cfg(feature = "tensorrt")]
    #[test]
    #[ignore]
    fn trt_session_matches_the_plain_session() {
        use crate::board::Board;
        use crate::board::rules::RulesEnum;
        use crate::mcts::get_left_moves;
        use crate::movegen::MoveGen;
        let path =
            std::env::var("TAFL_GRAPH_NET").unwrap_or_else(|_| "copenhagen11x11.onnx".into());
        let batch = 64;
        let (plain, size, planes) = NeuralNet::build_session_on(&path, 11, batch, false).unwrap();
        let (trt, _, _) = NeuralNet::build_session_on(&path, 11, batch, true).unwrap();
        let wrap = |session| NeuralNet {
            backend: Backend::Local(session),
            path: path.clone(),
            loaded_mtime: None,
            board_size: size,
            planes: planes.clone(),
            batch_size: batch,
        };
        let (mut plain, mut trt) = (wrap(plain), wrap(trt));
        let mut board = Board::new();
        board.set_rules(RulesEnum::Copenhagen11x11);
        board.setup_initial_position().unwrap();
        let mut mg = MoveGen::new();
        let mut inputs = Vec::new();
        let mut state = 12345u64;
        while inputs.len() < 100 {
            inputs.push(NnInput::from_board(&mut board, &mut mg, &planes));
            let moves = get_left_moves(&board, &mut mg);
            if moves.is_empty() || crate::terminal::check_terminal(&mut board).is_some() {
                board.setup_initial_position().unwrap();
                continue;
            }
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            board
                .make_move_simple(moves[(state >> 33) as usize % moves.len()])
                .unwrap();
        }
        let geom = board.precomputed().clone();
        let (mut worst_value, mut worst_prob, mut top_agree, mut total) = (0.0f32, 0.0f32, 0, 0);
        let softmax = |l: &[f32]| {
            let m = l.iter().copied().fold(f32::MIN, f32::max);
            let e: Vec<f32> = l.iter().map(|x| (x - m).exp()).collect();
            let z: f32 = e.iter().sum();
            e.into_iter().map(|x| x / z).collect::<Vec<f32>>()
        };
        for chunk in [&inputs[..64], &inputs[64..65], &inputs[65..100]] {
            let refs: Vec<&NnInput> = chunk.iter().collect();
            let a = plain.evaluate_batch(&refs, &geom);
            let b = trt.evaluate_batch(&refs, &geom);
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(&b) {
                // Compare legal-move probabilities (illegal logits are huge negatives
                // in neither, but their exact values do not matter to the search).
                let (p, q) = (softmax(&x.policy), softmax(&y.policy));
                let dp = p
                    .iter()
                    .zip(&q)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                worst_prob = worst_prob.max(dp);
                worst_value = worst_value.max((x.value - y.value).abs());
                let arg = |v: &[f32]| (0..v.len()).max_by(|&i, &j| v[i].total_cmp(&v[j])).unwrap();
                top_agree += (arg(&p) == arg(&q)) as usize;
                total += 1;
            }
        }
        println!(
            "{path}: max |prob diff| {worst_prob:.2e}, max |value diff| {worst_value:.2e}, top move agrees {top_agree}/{total}"
        );
        assert!(
            worst_prob < 2e-2 && worst_value < 2e-2,
            "{worst_prob} {worst_value}"
        );
        assert!(top_agree * 100 >= total * 95, "{top_agree}/{total}");
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
