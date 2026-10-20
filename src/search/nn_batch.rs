//! One ORT session shared by many game threads.
//!
//! A datagen process that plays K games in K threads hands every `evaluate_batch`
//! to this server instead of running its own inference. The server pools the
//! requests that arrive while it is busy (or within a short wait) into one tensor
//! of up to `max_batch` samples and runs them together, so the GPU sees a few
//! large batches a second instead of K streams of 8-sample calls.
//!
//! Numerically this is the same computation: every sample is an independent row
//! of the batch, so a position evaluated here gets the same policy/value it gets
//! from a private session (within kernel-selection rounding for a different shape).

use super::nn_common::NnOutput;
use super::nn_native::{NeuralNet, run_inference};
use crate::get_sample_size;
use ort::session::Session;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::{Duration, Instant};

/// Batches are padded to a multiple of this, so ORT only ever sees
/// `max_batch / BUCKET` distinct input shapes (each planned once at start-up).
pub const BUCKET: usize = 32;

struct Request {
    input: Vec<f32>,
    n: usize,
    reply: mpsc::SyncSender<Vec<NnOutput>>,
}

#[derive(Default)]
struct Queue {
    requests: Vec<Request>,
    samples: usize,
}

pub struct BatchServer {
    queue: Mutex<Queue>,
    arrived: Condvar,
    board_size: usize,
    max_batch: usize,
    wait: Duration,
    /// Threads currently playing; a batch is closed as soon as half of them have a
    /// request in, without waiting out `wait`.
    active: AtomicUsize,
    /// Batches run and samples served, for the throughput line in logs.
    batches: AtomicUsize,
    served: AtomicUsize,
    /// Successful net reloads (net_watch).
    reloads: AtomicUsize,
}

fn file_mtime(path: &str) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

impl BatchServer {
    /// Loads the net and starts the serving thread. `max_batch` is rounded up to a
    /// multiple of [`BUCKET`]; `wait` bounds how long a batch is held open for more
    /// requests. With `net_watch` the serving thread reloads the file when its
    /// mtime changes (checked between batches, at most once a second).
    pub fn start(
        path: &str,
        board_size: usize,
        max_batch: usize,
        wait: Duration,
        net_watch: bool,
    ) -> Result<Arc<Self>, String> {
        let max_batch = max_batch.max(1).div_ceil(BUCKET) * BUCKET;
        let (mut session, board_size) = NeuralNet::build_session(path, board_size, max_batch)?;
        warm_buckets(&mut session, board_size, max_batch);
        let server = Arc::new(Self {
            queue: Mutex::new(Queue::default()),
            arrived: Condvar::new(),
            board_size,
            max_batch,
            wait,
            active: AtomicUsize::new(0),
            batches: AtomicUsize::new(0),
            served: AtomicUsize::new(0),
            reloads: AtomicUsize::new(0),
        });
        let worker = Arc::clone(&server);
        let path = path.to_string();
        std::thread::Builder::new()
            .name("nn-batch".into())
            .spawn(move || worker.serve(session, path, net_watch))
            .map_err(|e| format!("cannot start nn batch thread: {e}"))?;
        Ok(server)
    }

    pub fn board_size(&self) -> usize {
        self.board_size
    }

    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Net reloads completed so far.
    pub fn reloads(&self) -> usize {
        self.reloads.load(Ordering::Relaxed)
    }

    /// (batches run, samples served) so far.
    pub fn stats(&self) -> (usize, usize) {
        (
            self.batches.load(Ordering::Relaxed),
            self.served.load(Ordering::Relaxed),
        )
    }

    /// A game thread announces itself; the server then closes a batch as soon as
    /// every active thread has handed in a request.
    pub fn client_started(&self) {
        self.active.fetch_add(1, Ordering::Relaxed);
    }

    pub fn client_stopped(&self) {
        self.active.fetch_sub(1, Ordering::Relaxed);
        // A batch may be waiting for this thread's request that will never come.
        self.arrived.notify_all();
    }

    /// Evaluates `n` samples already laid out as input planes. Blocks until the
    /// batch containing them has run.
    pub fn evaluate(&self, input: Vec<f32>, n: usize) -> Vec<NnOutput> {
        assert!(
            n <= self.max_batch,
            "request of {n} samples exceeds the server's max batch {}",
            self.max_batch
        );
        let (reply, rx) = mpsc::sync_channel(1);
        {
            let mut q = self.queue.lock().unwrap();
            q.requests.push(Request { input, n, reply });
            q.samples += n;
        }
        self.arrived.notify_all();
        rx.recv().expect("nn batch server thread is gone")
    }

    fn take_batch(&self) -> (Vec<Request>, usize) {
        let mut q = self.queue.lock().unwrap();
        while q.requests.is_empty() {
            q = self.arrived.wait(q).unwrap();
        }
        // Hold the batch open while it is short of both the size cap and a request
        // from half the active threads, but never longer than `wait`. Half, not all:
        // waiting for every thread makes the whole process march in lockstep (GPU
        // idle while threads select, threads idle while the GPU runs). With half,
        // one group's batch runs while the other group selects its next leaves.
        let deadline = Instant::now() + self.wait;
        loop {
            let active = self.active.load(Ordering::Relaxed);
            if q.samples >= self.max_batch || q.requests.len() >= active.div_ceil(2) {
                break;
            }
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            q = self.arrived.wait_timeout(q, deadline - now).unwrap().0;
        }
        // Whole requests only, up to the cap; the rest stays for the next batch.
        let mut taken = Vec::with_capacity(q.requests.len());
        let mut total = 0;
        let mut i = 0;
        while i < q.requests.len() {
            if total + q.requests[i].n <= self.max_batch {
                let r = q.requests.remove(i);
                total += r.n;
                taken.push(r);
            } else {
                i += 1;
            }
        }
        q.samples -= total;
        (taken, total)
    }

    fn serve(&self, mut session: Session, path: String, net_watch: bool) {
        let sample_size = get_sample_size(self.board_size);
        let mut loaded_mtime = file_mtime(&path);
        let mut last_check = Instant::now();
        // A reload (session build + warm-up, about a second) runs on a helper thread
        // so the game threads keep being served by the old net meanwhile; the new
        // session is swapped in between batches.
        let mut pending_reload: Option<mpsc::Receiver<Result<Session, String>>> = None;
        loop {
            let (taken, total) = self.take_batch();
            let padded = total.div_ceil(BUCKET) * BUCKET;
            let mut input = Vec::with_capacity(padded * sample_size);
            for r in &taken {
                input.extend_from_slice(&r.input);
            }
            let outputs = run_inference(&mut session, input, total, padded, self.board_size);
            let mut outputs = outputs.into_iter();
            for r in taken {
                let out: Vec<NnOutput> = outputs.by_ref().take(r.n).collect();
                // A client that gave up (thread ended) just drops its receiver.
                let _ = r.reply.send(out);
            }
            self.batches.fetch_add(1, Ordering::Relaxed);
            self.served.fetch_add(total, Ordering::Relaxed);

            if let Some(rx) = &pending_reload {
                match rx.try_recv() {
                    Ok(Ok(s)) => {
                        session = s;
                        pending_reload = None;
                        self.reloads.fetch_add(1, Ordering::Relaxed);
                        println!("[NN] Reloaded {path}");
                    }
                    Ok(Err(err)) => {
                        pending_reload = None;
                        println!("[NN] Reload failed: {err}, keeping old net");
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => pending_reload = None,
                }
            }
            if net_watch
                && pending_reload.is_none()
                && last_check.elapsed() >= Duration::from_secs(1)
            {
                last_check = Instant::now();
                let mtime = file_mtime(&path);
                if mtime.is_some() && mtime != loaded_mtime {
                    loaded_mtime = mtime;
                    let (tx, rx) = mpsc::channel();
                    let (path, board_size, max_batch) =
                        (path.clone(), self.board_size, self.max_batch);
                    std::thread::spawn(move || {
                        let built = match NeuralNet::build_session(&path, board_size, max_batch) {
                            Ok((mut s, size)) if size == board_size => {
                                warm_buckets(&mut s, size, max_batch);
                                Ok(s)
                            }
                            Ok((_, size)) => Err(format!("net is for a {size}x{size} board")),
                            Err(err) => Err(err),
                        };
                        let _ = tx.send(built);
                    });
                    pending_reload = Some(rx);
                }
            }
        }
    }
}

/// `build_session` warms up `max_batch` only; plan the smaller shapes too so the
/// first short batch does not stall on ORT's per-shape setup.
fn warm_buckets(session: &mut Session, board_size: usize, max_batch: usize) {
    let sample_size = get_sample_size(board_size);
    let mut b = BUCKET;
    while b < max_batch {
        let _ = run_inference(session, vec![0.0; b * sample_size], b, b, board_size);
        b += BUCKET;
    }
}

#[cfg(test)]
mod tests {
    use super::super::nn_common::build_input_data;
    use super::*;
    use crate::board::Board;
    use crate::board::position_export::BitPosition;
    use crate::board::rules::RulesEnum;

    fn fresh() -> Board {
        let mut board = Board::new();
        board.set_rules(RulesEnum::Copenhagen11x11);
        board.setup_initial_position().unwrap();
        board
    }

    /// `count` positions from random play (restarting after a terminal), plus a
    /// board to borrow the geometry from.
    fn positions(seed: u64, count: usize) -> (Vec<BitPosition>, Board) {
        use rand::{RngExt, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(seed);
        let mut movegen = crate::moves::movegen::MoveGen::new();
        let mut out = Vec::with_capacity(count);
        let mut b = fresh();
        while out.len() < count {
            let moves = crate::search::mcts::get_left_moves(&b, &mut movegen);
            if moves.is_empty() || crate::evaluation::terminal::check_terminal(&mut b).is_some() {
                b = fresh();
                continue;
            }
            let mv = moves[rng.random_range(0..moves.len())];
            b.make_move_simple(mv).unwrap();
            out.push(BitPosition::from_board(&b, 1));
        }
        (out, fresh())
    }

    // The pooled path gives each thread exactly what a private session gives it,
    // whatever the interleaving of requests.
    #[test]
    fn shared_server_matches_private_session() {
        let server = BatchServer::start(
            "copenhagen11x11.onnx",
            11,
            4 * 8,
            Duration::from_millis(2),
            false,
        )
        .unwrap();
        assert_eq!(server.max_batch(), 32);
        let mut local = NeuralNet::new("copenhagen11x11.onnx", 11, 8).unwrap();

        std::thread::scope(|scope| {
            for t in 0..4u64 {
                let server = Arc::clone(&server);
                scope.spawn(move || {
                    let (pos, board) = positions(100 + t, 40);
                    let geom = board.precomputed();
                    let mut shared = NeuralNet::shared(server.clone());
                    let mut expect = NeuralNet::new("copenhagen11x11.onnx", 11, 8).unwrap();
                    server.client_started();
                    for chunk in pos.chunks(1 + t as usize * 2) {
                        let refs: Vec<&BitPosition> = chunk.iter().collect();
                        let got = shared.evaluate_batch(&refs, geom);
                        let want = expect.evaluate_batch(&refs, geom);
                        assert_eq!(got.len(), want.len());
                        for (g, w) in got.iter().zip(&want) {
                            assert!(
                                (g.value - w.value).abs() < 1e-4,
                                "{} vs {}",
                                g.value,
                                w.value
                            );
                            assert_eq!(g.moves_left.is_some(), w.moves_left.is_some());
                            let d = g
                                .policy
                                .iter()
                                .zip(&w.policy)
                                .map(|(a, b)| (a - b).abs())
                                .fold(0.0f32, f32::max);
                            assert!(d < 1e-3, "policy differs by {d}");
                        }
                    }
                    server.client_stopped();
                });
            }
        });

        let (batches, served) = server.stats();
        assert_eq!(served, 160);
        assert!(
            batches < 160,
            "requests were not pooled at all: {batches} batches"
        );

        // And the server sees the same thing the local session sees for one sample.
        let (pos, board) = positions(7, 1);
        let geom = board.precomputed();
        let input = build_input_data(&[&pos[0]], geom);
        let a = server.evaluate(input, 1).pop().unwrap();
        let b = local.evaluate_position(&pos[0], geom);
        assert!((a.value - b.value).abs() < 1e-4);
    }

    // With net_watch the server picks up a newer file while it keeps serving, and
    // answers from the new session afterwards.
    #[test]
    fn reloads_in_the_background_while_serving() {
        let dir = std::env::temp_dir().join(format!("taflzero-srv-reload-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let copy = dir.join("net.onnx");
        std::fs::write(&copy, std::fs::read("copenhagen11x11.onnx").unwrap()).unwrap();
        let server = BatchServer::start(
            copy.to_str().unwrap(),
            11,
            8,
            Duration::from_millis(1),
            true,
        )
        .unwrap();
        let (pos, board) = positions(3, 4);
        let geom = board.precomputed();
        let mut shared = NeuralNet::shared(server.clone());
        server.client_started();
        let refs: Vec<&BitPosition> = pos.iter().collect();
        let before = shared.evaluate_batch(&refs, geom);

        // Same bytes, newer mtime: the next check (≥ 1 s after the last) starts a
        // background rebuild; keep evaluating until it has been swapped in.
        let f = std::fs::OpenOptions::new().write(true).open(&copy).unwrap();
        f.set_modified(std::time::SystemTime::now() + Duration::from_secs(5))
            .unwrap();
        drop(f);
        let t0 = Instant::now();
        while server.reloads() == 0 && t0.elapsed() < Duration::from_secs(60) {
            let _ = shared.evaluate_batch(&refs, geom);
            std::thread::sleep(Duration::from_millis(50));
        }
        assert_eq!(server.reloads(), 1, "net was not reloaded");
        let after = shared.evaluate_batch(&refs, geom);
        for (x, y) in before.iter().zip(&after) {
            assert!((x.value - y.value).abs() < 1e-4);
        }
        server.client_stopped();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
