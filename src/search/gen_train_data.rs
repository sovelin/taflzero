use crate::board::Board;
use crate::board::position_export::BitPosition;
use crate::board::rules::RulesEnum;
use crate::board::types::Piece;
use crate::board::types::Side;
use crate::mcts::export::{KING_CORNER_NONE, LegalMask, PendingSample, king_corner_index};
use crate::mcts::utils::move_to_policy_index;
use crate::mcts::{C_PUCT, GumbelConfig, MCTSConfig, MCTSTree, mcts_search};
use crate::movegen::MoveGen;
use crate::search::features::{PositionFeatures, finish_game, position_features, terminal_code};
use crate::search::nn::NeuralNet;
use crate::search_data::SearchData;
use crate::terminal::{check_terminal, get_terminal};
use crate::undo::UndoMove;
use rand::RngExt;
use rand::SeedableRng;
use rand::prelude::StdRng;
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::hash::{BuildHasher, Hasher};
use std::io::{BufWriter, Read, Write};
use std::sync::Mutex;

const CURRICULUM_TAIL: usize = 25;
const CURRICULUM_MIN_READY: usize = 100;

// ─── Curriculum buffer ────────────────────────────────────────────────────────

struct CurriculumBuffer {
    positions: VecDeque<BitPosition>,
    max_size: usize,
}

impl CurriculumBuffer {
    fn new(max_size: usize) -> Self {
        Self {
            positions: VecDeque::new(),
            max_size,
        }
    }

    fn push_game_tail(&mut self, history: &[BitPosition]) {
        let tail_start = history.len().saturating_sub(CURRICULUM_TAIL);
        for &bp in &history[tail_start..] {
            if self.positions.len() >= self.max_size {
                self.positions.pop_front();
            }
            self.positions.push_back(bp);
        }
    }

    fn sample(&self, rnd: &mut StdRng) -> Option<BitPosition> {
        if self.positions.is_empty() {
            return None;
        }
        let idx = rnd.random_range(0..self.positions.len());
        Some(self.positions[idx])
    }

    fn is_ready(&self) -> bool {
        self.positions.len() >= CURRICULUM_MIN_READY
    }

    fn load_from_file(&mut self, path: &str) {
        let Ok(mut f) = std::fs::File::open(path) else {
            return;
        };
        let bp_size = size_of::<BitPosition>();
        let mut buf = vec![0u8; bp_size];
        let mut loaded = 0usize;
        while f.read_exact(&mut buf).is_ok() {
            let bp: BitPosition = unsafe { std::ptr::read(buf.as_ptr() as *const BitPosition) };
            if self.positions.len() >= self.max_size {
                self.positions.pop_front();
            }
            self.positions.push_back(bp);
            loaded += 1;
        }
        if loaded > 0 {
            println!("[curriculum] Loaded {} positions from {}", loaded, path);
        }
    }

    fn trim_and_rewrite_file(&self, path: &str) {
        // Trim file to current buffer contents on startup (drop stale oldest)
        let Ok(mut f) = OpenOptions::new().write(true).truncate(true).open(path) else {
            // file doesn't exist yet — nothing to trim
            return;
        };
        let bp_size = size_of::<BitPosition>();
        for bp in &self.positions {
            let bytes = unsafe {
                std::slice::from_raw_parts(bp as *const BitPosition as *const u8, bp_size)
            };
            let _ = f.write_all(bytes);
        }
    }
}

// ─── Terminal type string ─────────────────────────────────────────────────────

// ─── Game loop ────────────────────────────────────────────────────────────────

/// A played game: its samples, the positions for the curriculum, the winner, the
/// terminal's gamelog name and the side-file records.
type PlayedGame = (
    Vec<PendingSample>,
    Vec<BitPosition>,
    Option<Side>,
    Option<&'static str>,
    Vec<u8>,
);

fn play_game(
    nn: &mut NeuralNet,
    search_data: &mut SearchData,
    start_board: Board,
    search_cfg: &SearchConfig,
    limits: &GameLimits,
    rng: &mut StdRng,
) -> PlayedGame {
    let mut board = start_board;

    // The side-file records (features.rs): one per sample, from the board as it stands
    // when the sample is taken, finished when the game ends.
    let mut feat_mg = MoveGen::new();
    let mut features: Vec<PositionFeatures> = vec![];
    let mut played: Vec<(u8, u8)> = vec![];
    let mut move_flags: Vec<u8> = vec![];
    let mut last_undo = UndoMove::new();

    let mut res: Vec<PendingSample> = vec![];
    let mut board_history: Vec<BitPosition> = vec![];

    // Full searches: Dirichlet noise + forced playouts, policy target written.
    // Cheap searches (playout cap randomization): no noise, policy target
    // flagged invalid — the position still contributes value/root_q data.
    let mut full_config = MCTSConfig::default_train();
    let mut cheap_config = MCTSConfig::default_train();
    cheap_config.dirichlet_alpha = 0.0;
    cheap_config.dirichlet_epsilon = 0.0;
    cheap_config.forced_playouts_k = 0.0;
    full_config.batch_size = search_cfg.search_batch.max(1);
    cheap_config.batch_size = search_cfg.search_batch.max(1);
    if let Some(considered) = search_cfg.gumbel {
        let g = GumbelConfig {
            considered,
            ..GumbelConfig::default()
        };
        for c in [&mut full_config, &mut cheap_config] {
            c.gumbel = Some(g);
            c.dirichlet_alpha = 0.0;
            c.dirichlet_epsilon = 0.0;
            c.forced_playouts_k = 0.0;
        }
    }

    // Per sample: the surprise byte and FLAGS2 of its side-file record.
    let mut surprise: Vec<u8> = vec![];
    let mut flags2: Vec<u8> = vec![];
    // Soft resignation (KataGo's reduceVisits): the last root values in the attackers'
    // view; once the last `soft_resign_plies` all lie beyond the threshold on one side,
    // full searches shrink toward the cheap size and the samples' training weight falls,
    // in proportion to how far past the threshold the game is. Never ends a game.
    let mut recent_atk_q: std::collections::VecDeque<f32> = std::collections::VecDeque::new();

    let game_result;
    let terminal_str: Option<&'static str>;
    let mut move_number: usize = 0;
    let mut mcts_tree = MCTSTree::new();
    let mut no_capture_counter = 0;

    loop {
        // Temperature schedule: 1.0 for the opening, then linear decay to a
        // small floor so late-game move selection stays mostly greedy but can
        // still deviate when candidates are nearly equal.
        let temperature = if move_number < 8 {
            1.0
        } else if move_number < 22 {
            1.0 - 0.85 * ((move_number - 8) as f32 / 14.0)
        } else {
            0.15
        };
        full_config.temperature = temperature;
        cheap_config.temperature = temperature;

        let is_full = search_cfg.full_prob >= 1.0 || rng.random::<f64>() < search_cfg.full_prob;
        let reduction = match search_cfg.soft_resign {
            Some(threshold) => {
                soft_resign_reduction(&recent_atk_q, search_cfg.soft_resign_plies, threshold)
            }
            None => 0.0,
        };
        let (config, nodes) = if is_full {
            // A reduced full search is still a full search (policy target written), just
            // smaller: from full_nodes at no reduction to cheap_nodes at a full one.
            let full = search_cfg.full_nodes as f32;
            let cheap = search_cfg.cheap_nodes as f32;
            let n = (full + reduction * (cheap - full)).round() as u64;
            (&full_config, n.max(search_cfg.cheap_nodes).max(1))
        } else {
            (&cheap_config, search_cfg.cheap_nodes)
        };

        // Sequential halving assumes a fresh root: no tree reuse between moves.
        if search_cfg.gumbel.is_some() {
            mcts_tree = MCTSTree::new();
        }

        let mv = mcts_search(
            &mut board,
            &mut mcts_tree,
            nn,
            search_data,
            None,
            Some(nodes),
            config,
            None,
        );
        move_number += 1;

        if let Some(mv) = mv {
            let prune_k = if is_full {
                config.forced_playouts_k
            } else {
                0.0
            };
            let mut sample = mcts_tree.make_pending_sample(&board, prune_k, C_PUCT);
            sample.set_policy_valid(is_full);
            res.push(sample);
            surprise.push(if is_full {
                crate::search::features::surprise_byte(mcts_tree.policy_surprise())
            } else {
                0
            });
            flags2.push(crate::search::features::reduction_byte(reduction));
            if search_cfg.soft_resign.is_some() {
                let q = mcts_tree.root_q();
                let atk_q = if board.side_to_move == Side::ATTACKERS {
                    q
                } else {
                    -q
                };
                recent_atk_q.push_back(atk_q);
                while recent_atk_q.len() > search_cfg.soft_resign_plies.max(1) {
                    recent_atk_q.pop_front();
                }
            }

            let rep = board.rep_table.get(&board.zobrist).copied().unwrap_or(1);
            board_history.push(BitPosition::from_board(&board, rep));

            features.push(position_features(&mut board, &mut feat_mg));
            let king_move = board.board[mv.from()] == Piece::KING;

            let pieces_count = board.attackers_count + board.defenders_count;
            board
                .make_move(mv, &mut last_undo)
                .expect("Make move failed");
            let new_pieces_count = board.attackers_count + board.defenders_count;
            played.push((mv.from() as u8, mv.to() as u8));
            move_flags.push(
                if king_move {
                    crate::search::features::FEAT_KING_MOVE
                } else {
                    0
                } | if last_undo.captured_pieces_count > 0 {
                    crate::search::features::FEAT_CAPTURE
                } else {
                    0
                },
            );

            if new_pieces_count != pieces_count {
                no_capture_counter = 0;
            } else {
                no_capture_counter += 1;
            }

            // Terminals first: a game that ends exactly on the limit is still decided.
            if let Some(terminal) = get_terminal(&mut board) {
                println!("{}", board);
                terminal_str = Some(terminal.name());
                // A third repetition is an attacker win under the rules and is stored
                // as one: a defence that only shuffles must learn that it loses.
                game_result = Some(
                    check_terminal(&mut board)
                        .expect("get_terminal and check_terminal disagree on a terminal position"),
                );
                break;
            }

            // Undecided: the outcome is unknown, the samples are still written
            // (policy and root_q targets stay valid; the value target is masked).
            if no_capture_counter >= limits.max_no_capture || move_number >= limits.max_plies {
                game_result = None;
                terminal_str = Some(if no_capture_counter >= limits.max_no_capture {
                    "undecided_nocapture"
                } else {
                    "undecided_plies"
                });
                break;
            }
        } else {
            game_result = if board.side_to_move == Side::ATTACKERS {
                terminal_str = Some("def_no_moves");
                println!("{}", board);
                Some(Side::DEFENDERS)
            } else {
                terminal_str = Some("atk_no_moves");
                println!("{}", board);
                Some(Side::ATTACKERS)
            };
            break;
        }
    }

    // Backfill game-outcome targets: value, escape corner, last-move flag.
    let king_corner = if terminal_str == Some("def_corner") {
        king_corner_index(board.king_sq)
    } else {
        KING_CORNER_NONE
    };
    for sample in res.iter_mut() {
        sample.set_value_from_result(game_result);
        sample.set_king_corner(king_corner);
    }
    if let Some(last) = res.last_mut() {
        last.set_last_of_game();
    }

    // Side-file records. The last sample's move is read off the result exactly as
    // the replay tool (`taflzero --features`) does, so both write the same bytes;
    // only how the game ended is taken from what this game knows.
    let mut feat_records =
        Vec::with_capacity(res.len() * crate::search::features::FEAT_RECORD_BYTES);
    if let Some(last) = res.last() {
        // Back to the last sample's position: the last move made is its move.
        board
            .unmake_move(&mut last_undo)
            .expect("unmake of the last move failed");
        let n = res.len();
        let mut flags = move_flags;
        flags[n - 1] = 0;
        finish_game(
            &features,
            &played[..n - 1],
            flags,
            &mut board,
            last.flags(),
            last.value(),
            terminal_str.map(terminal_code),
            &mut feat_records,
        );
        crate::search::features::set_training_bytes(&mut feat_records, &surprise, &flags2);
    }

    (res, board_history, game_result, terminal_str, feat_records)
}

/// KataGo's soft-resignation reduction (play.cpp, reduceVisits) for the next search: over
/// the last `plies` root values (attackers' view, newest last) take the least extreme one
/// on a common side; if it lies past `threshold`, the reduction is
/// ((value - threshold) / (1 - threshold))^2, else 0. So 0 at the threshold, 0.25 halfway
/// to 1, and 1 only for a game that is all but over: a gradual cut, no state to leave.
fn soft_resign_reduction(
    recent_atk_q: &std::collections::VecDeque<f32>,
    plies: usize,
    threshold: f32,
) -> f32 {
    let plies = plies.max(1);
    if recent_atk_q.len() < plies || threshold >= 1.0 {
        return 0.0;
    }
    let last = recent_atk_q.iter().rev().take(plies);
    let (lo, hi) = last.fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &q| {
        (lo.min(q), hi.max(q))
    });
    let most_extreme = lo.max(-hi).min(1.0);
    let through = most_extreme - threshold;
    if through <= 0.0 {
        return 0.0;
    }
    let prop = (through / (1.0 - threshold)).clamp(0.0, 1.0);
    prop * prop
}

// ─── Main datagen loop ────────────────────────────────────────────────────────

/// Per-move search budget (playout cap randomization).
#[derive(Clone)]
pub struct SearchConfig {
    /// Nodes for full searches (policy target written).
    pub full_nodes: u64,
    /// Nodes for cheap searches (policy target flagged invalid).
    pub cheap_nodes: u64,
    /// Probability a move gets a full search. 1.0 = every move full (no PCR).
    pub full_prob: f64,
    /// Gumbel AlphaZero at the root with this many considered actions; `None` = PUCT
    /// with Dirichlet noise and forced playouts.
    pub gumbel: Option<usize>,
    /// Leaves selected per net call inside one search. 1 = plain MCTS, no virtual
    /// loss; the GPU batch then comes from many games (`--threads`) instead.
    pub search_batch: usize,
    /// Soft resignation threshold on |root value| (KataGo's reduceVisits); `None` = off.
    pub soft_resign: Option<f32>,
    /// Consecutive plies beyond the threshold that start the reduced state.
    pub soft_resign_plies: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            full_nodes: 400,
            cheap_nodes: 100,
            full_prob: 1.0,
            gumbel: None,
            search_batch: MCTSConfig::default_train().batch_size,
            soft_resign: None,
            soft_resign_plies: 3,
        }
    }
}

/// Settings a running trainer cannot change on its workers' command line: a text
/// file next to the data files (`worker.overrides`, `key=value` per line, `#`
/// comments) read by the engine at start, so a worker restart is enough to try them.
#[derive(Default, Debug, PartialEq)]
pub struct WorkerOverrides {
    pub threads: Option<usize>,
    pub search_batch: Option<usize>,
    pub ramp_seconds: Option<usize>,
}

impl WorkerOverrides {
    pub const FILE: &'static str = "worker.overrides";

    pub fn parse(text: &str) -> Result<Self, String> {
        let mut out = Self::default();
        for (i, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| format!("line {}: expected key=value, got '{line}'", i + 1))?;
            let value: usize = value
                .trim()
                .parse()
                .map_err(|_| format!("line {}: '{}' is not a number", i + 1, value.trim()))?;
            match key.trim() {
                "threads" => out.threads = Some(value.max(1)),
                "search_batch" => out.search_batch = Some(value.max(1)),
                "ramp_seconds" => out.ramp_seconds = Some(value),
                other => return Err(format!("line {}: unknown key '{other}'", i + 1)),
            }
        }
        Ok(out)
    }

    /// The overrides file next to a datagen output file, if there is one.
    pub fn next_to(datagen_path: &str) -> Option<(String, Result<Self, String>)> {
        let dir = std::path::Path::new(datagen_path).parent()?;
        let file = dir.join(Self::FILE);
        let text = std::fs::read_to_string(&file).ok()?;
        Some((file.to_string_lossy().into_owned(), Self::parse(&text)))
    }
}

/// When a game is cut off as undecided.
#[derive(Clone)]
pub struct GameLimits {
    /// Plies after which the game is abandoned.
    pub max_plies: usize,
    /// Plies without a capture after which the game is abandoned.
    pub max_no_capture: usize,
}

impl Default for GameLimits {
    fn default() -> Self {
        Self {
            max_plies: 400,
            max_no_capture: 400,
        }
    }
}

/// Controls for a worker that runs unattended next to a trainer.
#[derive(Default, Clone)]
pub struct DaemonConfig {
    /// Reload the net before a game when the file's mtime changed.
    pub net_watch: bool,
    /// Finish the current game and exit when this file exists.
    pub stop_file: Option<String>,
    /// Do not start a new game while this file exists.
    pub pause_file: Option<String>,
    /// Log prefix only.
    pub worker_id: Option<usize>,
    /// Game thread inside a `--threads` worker; log prefix only.
    pub thread: Option<usize>,
}

#[derive(Clone)]
pub struct DatagenConfig {
    pub curriculum_fraction: f64,
    pub curriculum_path: Option<String>,
    pub curriculum_max_size: usize,
    pub search: SearchConfig,
    pub limits: GameLimits,
    pub daemon: DaemonConfig,
}

impl Default for DatagenConfig {
    fn default() -> Self {
        Self {
            curriculum_fraction: 0.0,
            curriculum_path: None,
            curriculum_max_size: 50_000,
            search: SearchConfig::default(),
            limits: GameLimits::default(),
            daemon: DaemonConfig::default(),
        }
    }
}

/// The sample and gamelog files of a datagen run. Game threads share one sink:
/// each game is written under the lock in a single `write_all` + flush, so the
/// file is always a sequence of whole games whatever the thread interleaving.
pub struct DatagenSink {
    writer: Mutex<GameFiles>,
    log: Mutex<BufWriter<std::fs::File>>,
}

/// The sample file and its feature side file (`workerN.bin` -> `workerN.feat`), written
/// under one lock: a game's samples first, then its records, each flushed, so the side
/// file lags the sample file by at most the game being written.
struct GameFiles {
    bin: BufWriter<std::fs::File>,
    feat: Option<BufWriter<std::fs::File>>,
}

/// `workerN.bin` -> `workerN.feat`.
pub fn feature_path(output_path: &str) -> std::path::PathBuf {
    std::path::Path::new(output_path).with_extension("feat")
}

impl DatagenSink {
    pub fn open(output_path: &str, log_path: &str, variant: RulesEnum) -> Self {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(output_path)
            .expect("Could not open output file");
        let log_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .expect("Could not open gamelog file");
        let feat = open_feature_file(output_path, variant);
        Self {
            writer: Mutex::new(GameFiles {
                bin: BufWriter::new(file),
                feat,
            }),
            log: Mutex::new(BufWriter::new(log_file)),
        }
    }
}

/// Opens the side file for appending, after making it cover every sample already in the
/// sample file: a worker that died between its two writes left the last game without
/// records, and the replay tool's code fills them in. A sample file with data but no side
/// file (data from an engine before side files) gets none: its records could only come
/// from replaying the whole file (`taflzero --features`), and the trainer reads such a
/// file as base-only.
#[cfg(not(target_arch = "wasm32"))]
fn open_feature_file(output_path: &str, variant: RulesEnum) -> Option<BufWriter<std::fs::File>> {
    let bin = std::path::Path::new(output_path);
    let path = feature_path(output_path);
    let bin_len = std::fs::metadata(bin).map(|m| m.len()).unwrap_or(0);
    let feat_len = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    if feat_len == 0 && bin_len > 0 {
        println!(
            "[datagen] {} has samples but no side file; not writing {} (make one with taflzero --features)",
            bin.display(),
            path.display()
        );
        return None;
    }
    if feat_len == 0 {
        std::fs::write(&path, crate::search::features::feature_header())
            .expect("Could not create feature file");
    } else {
        match crate::search::feature_dump::catch_up_features(bin, &path, variant) {
            Ok(0) => {}
            Ok(n) => println!("[datagen] {}: {n} missing records replayed", path.display()),
            Err(err) => {
                println!("[datagen] {}: {err}; not writing it", path.display());
                return None;
            }
        }
    }
    let file = OpenOptions::new()
        .append(true)
        .open(&path)
        .expect("Could not open feature file");
    Some(BufWriter::new(file))
}

#[cfg(target_arch = "wasm32")]
fn open_feature_file(_output_path: &str, _variant: RulesEnum) -> Option<BufWriter<std::fs::File>> {
    None
}

fn stop_requested(daemon: &DaemonConfig) -> bool {
    daemon
        .stop_file
        .as_ref()
        .is_some_and(|p| std::path::Path::new(p).exists())
}

pub fn gen_train_data(
    output_path: &str,
    log_path: &str,
    nn: &mut NeuralNet,
    game_limit: Option<usize>,
    variant: RulesEnum,
    cfg: DatagenConfig,
) {
    let sink = DatagenSink::open(output_path, log_path, variant);
    gen_train_data_into(&sink, nn, game_limit, variant, cfg);
}

/// One game loop writing into a (possibly shared) sink; `--threads N` runs N of
/// these over one sink and one batching net.
pub fn gen_train_data_into(
    sink: &DatagenSink,
    nn: &mut NeuralNet,
    game_limit: Option<usize>,
    variant: RulesEnum,
    cfg: DatagenConfig,
) {
    let mut search_data = SearchData::new();
    let mut rng = StdRng::seed_from_u64(std::hash::RandomState::new().build_hasher().finish());

    // ── Curriculum buffer setup ──
    let use_curriculum = cfg.curriculum_fraction > 0.0;
    let mut curriculum = CurriculumBuffer::new(cfg.curriculum_max_size);

    if use_curriculum && let Some(ref path) = cfg.curriculum_path {
        curriculum.load_from_file(path);
        curriculum.trim_and_rewrite_file(path);
    }

    // Append file handle for curriculum (opened lazily)
    let mut curriculum_file: Option<std::fs::File> = None;
    if use_curriculum && let Some(ref path) = cfg.curriculum_path {
        curriculum_file = OpenOptions::new().create(true).append(true).open(path).ok();
    }

    let bp_size = size_of::<BitPosition>();

    let mut positions_generated = 0usize;
    let mut games_played = 0usize;
    let mut attacker_wins_saved = 0usize;
    let mut defender_wins_saved = 0usize;
    let mut undecided_saved = 0usize;
    let tag = match (cfg.daemon.worker_id, cfg.daemon.thread) {
        (Some(id), Some(t)) => format!("[W{id}.{t}] "),
        (Some(id), None) => format!("[W{id}] "),
        (None, Some(t)) => format!("[T{t}] "),
        (None, None) => String::new(),
    };
    let mut curriculum_games = 0usize;
    let mut curriculum_decisive = 0usize;

    loop {
        if let Some(limit) = game_limit
            && positions_generated >= limit
        {
            println!(
                "{tag}Datagen finished: generated {} positions",
                positions_generated
            );
            break;
        }

        if stop_requested(&cfg.daemon) {
            println!(
                "{tag}Stop file present, exiting after {} games / {} positions",
                games_played, positions_generated
            );
            break;
        }

        if let Some(pause) = &cfg.daemon.pause_file {
            let mut announced = false;
            while std::path::Path::new(pause).exists() && !stop_requested(&cfg.daemon) {
                if !announced {
                    println!("{tag}Paused (pause file present)");
                    announced = true;
                }
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
            if stop_requested(&cfg.daemon) {
                continue;
            }
        }

        if cfg.daemon.net_watch {
            match nn.reload_if_changed() {
                Ok(true) => println!("{tag}[NN] Reloaded {}", nn.path()),
                Ok(false) => {}
                Err(err) => println!("{tag}[NN] Reload failed: {err}, keeping old net"),
            }
        }

        // Pick start position
        let is_curriculum_game = use_curriculum
            && curriculum.is_ready()
            && rng.random::<f64>() < cfg.curriculum_fraction;

        let start_board = if is_curriculum_game {
            let bp = curriculum.sample(&mut rng).unwrap();
            bp.to_board(variant)
        } else {
            let mut b = Board::new();
            b.set_rules(variant);
            b.setup_initial_position()
                .expect("Setup initial position failed");
            b
        };

        let (res, board_history, game_result, terminal_str, feat_records) = play_game(
            nn,
            &mut search_data,
            start_board,
            &cfg.search,
            &cfg.limits,
            &mut rng,
        );

        games_played += 1;
        if is_curriculum_game {
            curriculum_games += 1;
            if game_result.is_some() {
                curriculum_decisive += 1;
            }
        }

        match game_result {
            Some(Side::ATTACKERS) => attacker_wins_saved += 1,
            Some(Side::DEFENDERS) => defender_wins_saved += 1,
            None => undecided_saved += 1,
        }

        let total_decisive = attacker_wins_saved + defender_wins_saved;
        let atk_pct = if total_decisive > 0 {
            attacker_wins_saved as f64 / total_decisive as f64 * 100.0
        } else {
            0.0
        };
        let avg_game_len = if games_played > 0 {
            positions_generated as f64 / games_played as f64
        } else {
            0.0
        };
        let result_str = match game_result {
            Some(Side::ATTACKERS) => "ATK WIN",
            Some(Side::DEFENDERS) => "DEF WIN",
            None => "UNDECIDED",
        };
        let cur_tag = if is_curriculum_game { " [CUR]" } else { "" };
        println!(
            "{}{}{} | game #{} ({} samples) | atk={} def={} undecided={} | atk%={:.1}% | avg_len={:.1} | positions={}{}",
            tag,
            result_str,
            cur_tag,
            games_played,
            res.len(),
            attacker_wins_saved,
            defender_wins_saved,
            undecided_saved,
            atk_pct,
            avg_game_len,
            positions_generated,
            if curriculum_games > 0 {
                format!(
                    " | cur_decisive={}/{}",
                    curriculum_decisive, curriculum_games
                )
            } else {
                String::new()
            }
        );

        // Gamelog
        if let Some(t) = terminal_str {
            let mut log_writer = sink.log.lock().unwrap();
            writeln!(log_writer, "{},{}", t, res.len()).expect("Cannot write gamelog");
            log_writer.flush().expect("Cannot flush gamelog");
        }

        // Every game is written, undecided ones included (their value target is
        // masked by FLAG_RESULT_UNKNOWN). The whole game goes out in one write and is
        // flushed at once, so a reader tailing the file only ever sees a truncated
        // tail, never a torn sample in the middle.
        positions_generated += res.len();
        {
            let mut game_bytes: Vec<u8> = Vec::with_capacity(res.len() * 720);
            for sample in &res {
                sample
                    .write_to(&mut game_bytes)
                    .expect("Cannot serialize sample");
            }
            let mut files = sink.writer.lock().unwrap();
            files.bin.write_all(&game_bytes).expect("Cannot write game");
            files.bin.flush().expect("Cannot flush game");
            if let Some(feat) = files.feat.as_mut() {
                feat.write_all(&feat_records)
                    .expect("Cannot write feature records");
                feat.flush().expect("Cannot flush feature records");
            }
        }

        if game_result.is_some() {
            // Feed decisive game tail into curriculum buffer + file
            if use_curriculum {
                curriculum.push_game_tail(&board_history);
                if let Some(ref mut cf) = curriculum_file {
                    let tail_start = board_history.len().saturating_sub(CURRICULUM_TAIL);
                    for bp in &board_history[tail_start..] {
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                bp as *const BitPosition as *const u8,
                                bp_size,
                            )
                        };
                        let _ = cf.write_all(bytes);
                    }
                }
            }
        }
    }
}

// ─── Dump single sample (testing) ────────────────────────────────────────────

pub fn dump_single_sample(output_path: &str) {
    let mut board = Board::new();
    board
        .setup_initial_position()
        .expect("Setup initial position failed");

    let mut move_gen = MoveGen::new();
    move_gen.generate_moves(&board);
    if move_gen.count == 0 {
        panic!("No legal moves from initial position");
    }

    let first_mv = move_gen.moves[0];
    let move_index = move_to_policy_index(first_mv, board.board_size());

    let mut legal_mask = LegalMask::new(board.board_size());
    for i in 0..move_gen.count {
        let mv = move_gen.moves[i];
        let idx = move_to_policy_index(mv, board.board_size());
        legal_mask.set(idx as usize);
    }

    let policy = vec![(move_index, 7)];
    let sample =
        PendingSample::from_manual(BitPosition::from_board(&board, 1), legal_mask, policy, 1, 0);

    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(output_path)
        .expect("Could not open output file");
    let mut writer = BufWriter::new(file);
    sample.write_to(&mut writer).expect("Cannot write sample");

    println!(
        "DUMP_SAMPLE index={} legal_moves={}",
        move_index, move_gen.count
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcts::export::{FLAG_LAST_OF_GAME, FLAG_RESULT_UNKNOWN};
    use crate::terminal::TerminalType;

    fn play_from_start(
        limits: GameLimits,
    ) -> (Vec<PendingSample>, Option<Side>, Option<&'static str>) {
        let mut nn = NeuralNet::new("copenhagen11x11.onnx", 11, 8).unwrap();
        let mut search_data = SearchData::new();
        let mut rng = StdRng::seed_from_u64(7);
        let mut board = Board::new();
        board.set_rules(RulesEnum::Copenhagen11x11);
        board.setup_initial_position().unwrap();
        let search = SearchConfig {
            full_nodes: 4,
            cheap_nodes: 2,
            full_prob: 1.0,
            gumbel: None,
            ..SearchConfig::default()
        };
        let (res, _, result, term, _) =
            play_game(&mut nn, &mut search_data, board, &search, &limits, &mut rng);
        (res, result, term)
    }

    #[test]
    fn soft_resign_reduction_follows_katago() {
        let q = |v: &[f32]| {
            v.iter()
                .copied()
                .collect::<std::collections::VecDeque<f32>>()
        };
        let r = |v: &[f32]| soft_resign_reduction(&q(v), 3, 0.9);
        assert_eq!(r(&[0.95, 0.95]), 0.0); // fewer than N values
        assert_eq!(r(&[0.9, 0.95, 0.99]), 0.0); // the least extreme is at the threshold
        assert!((r(&[0.95, 0.97, 0.99]) - 0.25).abs() < 1e-5);
        assert!((r(&[-0.98, -0.99, -1.0]) - 0.64).abs() < 1e-5);
        assert!((r(&[1.0, 1.0, 1.0]) - 1.0).abs() < 1e-6);
        assert_eq!(r(&[0.95, -0.95, 0.99]), 0.0); // sides disagree
        assert_eq!(r(&[0.95, 0.5, 0.99]), 0.0);
        // only the last N count
        assert!((soft_resign_reduction(&q(&[0.0, 0.95, 0.95, 0.95]), 3, 0.9) - 0.25).abs() < 1e-5);
        use crate::search::features::{FEAT2_REDUCTION, reduction_byte};
        assert_eq!(reduction_byte(0.0), 0);
        assert_eq!(reduction_byte(1.0), FEAT2_REDUCTION | 127);
        assert_eq!(reduction_byte(0.25), FEAT2_REDUCTION | 32);
        assert_eq!(reduction_byte(1e-6), FEAT2_REDUCTION | 1);
    }

    #[test]
    fn surprise_byte_quantizes_and_caps() {
        use crate::search::features::surprise_byte;
        assert_eq!(surprise_byte(0.0), 0);
        assert_eq!(surprise_byte(-1.0), 0);
        assert_eq!(surprise_byte(f32::NAN), 0);
        assert_eq!(surprise_byte(1.0), 64);
        assert_eq!(surprise_byte(4.0), 255);
        assert_eq!(surprise_byte(9.0), 255);
    }

    // With a threshold every position passes, the reduction starts after the first
    // `plies` searches: from then on the samples carry a reduction byte, every search is
    // still a full one (full_prob 1, a reduced full search keeps its policy target and
    // surprise), and the game is played to its end (here the ply limit).
    #[test]
    fn soft_resign_reduces_searches_and_never_ends_the_game() {
        use crate::search::features::{FEAT_RECORD_BYTES, FEAT2_REDUCTION, offset};
        let mut nn = NeuralNet::new("copenhagen11x11.onnx", 11, 8).unwrap();
        let mut search_data = SearchData::new();
        let mut rng = StdRng::seed_from_u64(7);
        let mut board = Board::new();
        board.set_rules(RulesEnum::Copenhagen11x11);
        board.setup_initial_position().unwrap();
        let search = SearchConfig {
            full_nodes: 8,
            cheap_nodes: 2,
            full_prob: 1.0,
            gumbel: Some(4),
            soft_resign: Some(1e-6),
            soft_resign_plies: 2,
            ..SearchConfig::default()
        };
        let limits = GameLimits {
            max_plies: 12,
            max_no_capture: 400,
        };
        let (res, _, _, term, feat) =
            play_game(&mut nn, &mut search_data, board, &search, &limits, &mut rng);
        assert_eq!(feat.len(), res.len() * FEAT_RECORD_BYTES);
        let flags2: Vec<u8> = (0..res.len())
            .map(|i| feat[i * FEAT_RECORD_BYTES + offset::FLAGS2])
            .collect();
        let surprise: Vec<u8> = (0..res.len())
            .map(|i| feat[i * FEAT_RECORD_BYTES + offset::SURPRISE])
            .collect();
        // A root value of exactly 0 does not pass 1e-6; the opening rarely gives that,
        // so expect a reduction within the first few plies.
        let first = flags2
            .iter()
            .position(|&f| f & FEAT2_REDUCTION != 0)
            .expect("the game was never reduced");
        assert!(first >= 2, "reduced before 2 searches: {flags2:?}");
        for (i, s) in res.iter().enumerate() {
            let full = s.flags() & crate::mcts::export::FLAG_POLICY_VALID != 0;
            assert!(full, "sample {i}: every search is a full one here");
        }
        assert!(surprise.iter().any(|&b| b > 0), "no surprise: {surprise:?}");
        assert_eq!(
            term,
            Some("undecided_plies"),
            "the game must run to the limit"
        );
        assert_eq!(res.len(), 12);
    }

    // A game cut off by the ply limit is written as undecided: every sample carries
    // FLAG_RESULT_UNKNOWN with value 0, and the last one still closes the game.
    #[test]
    fn ply_limit_writes_undecided_game() {
        let (res, result, term) = play_from_start(GameLimits {
            max_plies: 6,
            max_no_capture: 400,
        });
        assert_eq!(result, None);
        assert_eq!(term, Some("undecided_plies"));
        assert_eq!(res.len(), 6);
        for s in &res {
            assert_ne!(s.flags() & FLAG_RESULT_UNKNOWN, 0);
            assert_eq!(s.value(), 0);
        }
        assert_ne!(res.last().unwrap().flags() & FLAG_LAST_OF_GAME, 0);
        assert_eq!(res[0].flags() & FLAG_LAST_OF_GAME, 0);
    }

    // The unknown bit is only ever set for games without a winner.
    #[test]
    fn decided_result_clears_unknown_flag() {
        let board = {
            let mut b = Board::new();
            b.set_rules(RulesEnum::Copenhagen11x11);
            b.setup_initial_position().unwrap();
            b
        };
        let mut s = PendingSample::from_manual(
            BitPosition::from_board(&board, 1),
            LegalMask::new(11),
            vec![],
            0,
            0,
        );
        s.set_value_from_result(None);
        assert_ne!(s.flags() & FLAG_RESULT_UNKNOWN, 0);
        s.set_value_from_result(Some(Side::ATTACKERS));
        assert_eq!(s.flags() & FLAG_RESULT_UNKNOWN, 0);
        assert_eq!(s.value(), 1);
        s.set_value_from_result(Some(Side::DEFENDERS));
        assert_eq!(s.value(), -1);
    }

    // A third repetition is a terminal with the attackers as winner; self-play stores
    // it as such instead of dropping the game.
    #[test]
    fn threefold_repetition_is_an_attacker_win() {
        let mut board = Board::new();
        board.set_rules(RulesEnum::Copenhagen11x11);
        board.setup_initial_position().unwrap();
        board.rep_table.insert(board.zobrist, 3);
        assert_eq!(
            get_terminal(&mut board),
            Some(TerminalType::ThreefoldRepetition)
        );
        assert_eq!(check_terminal(&mut board), Some(Side::ATTACKERS));
        assert_eq!(TerminalType::ThreefoldRepetition.name(), "atk_threefold");
    }

    // Daemon mode: an unlimited run ends cleanly once the stop file appears, and the
    // data file holds only whole samples.
    #[test]
    fn stop_file_ends_unlimited_run_with_whole_samples() {
        let dir = std::env::temp_dir().join(format!("taflzero-stop-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("w0.bin");
        let log = dir.join("w0.gamelog");
        let stop = dir.join("STOP");
        let _ = std::fs::remove_file(&stop);

        let mut nn = NeuralNet::new("copenhagen11x11.onnx", 11, 8).unwrap();
        let cfg = DatagenConfig {
            search: SearchConfig {
                full_nodes: 2,
                cheap_nodes: 2,
                full_prob: 1.0,
                gumbel: None,
                ..SearchConfig::default()
            },
            limits: GameLimits {
                max_plies: 5,
                max_no_capture: 400,
            },
            daemon: DaemonConfig {
                net_watch: true,
                stop_file: Some(stop.to_string_lossy().into_owned()),
                pause_file: None,
                worker_id: Some(0),
                thread: None,
            },
            ..DatagenConfig::default()
        };
        // Stop appears while the first games are being played.
        let stop_path = stop.clone();
        let t = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(300));
            std::fs::write(&stop_path, b"").unwrap();
        });
        gen_train_data(
            out.to_str().unwrap(),
            log.to_str().unwrap(),
            &mut nn,
            None,
            RulesEnum::Copenhagen11x11,
            cfg,
        );
        t.join().unwrap();

        let bytes = std::fs::read(&out).unwrap();
        assert!(!bytes.is_empty());
        // Walk the v2 layout: 50 + 605 + 2 + 4*len + 4 per sample.
        let mask = crate::mcts::export::legal_mask_bytes(11);
        let mut pos = 0usize;
        let mut samples = 0usize;
        while pos < bytes.len() {
            let len_at = pos + 50 + mask;
            assert!(len_at + 2 <= bytes.len(), "truncated sample header");
            let plen = u16::from_le_bytes([bytes[len_at], bytes[len_at + 1]]) as usize;
            pos = len_at + 2 + plen * 4 + 4;
            samples += 1;
        }
        assert_eq!(pos, bytes.len(), "truncated trailing sample");
        assert!(samples >= 5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Number of whole samples in a v2 data file; panics on a torn sample.
    fn count_samples(bytes: &[u8]) -> (usize, usize) {
        let mask = crate::mcts::export::legal_mask_bytes(11);
        let mut pos = 0usize;
        let mut samples = 0usize;
        let mut games = 0usize;
        while pos < bytes.len() {
            let len_at = pos + 50 + mask;
            assert!(len_at + 2 <= bytes.len(), "truncated sample header");
            let plen = u16::from_le_bytes([bytes[len_at], bytes[len_at + 1]]) as usize;
            let flags = bytes[len_at + 2 + plen * 4 + 2];
            if flags & FLAG_LAST_OF_GAME != 0 {
                games += 1;
            }
            pos = len_at + 2 + plen * 4 + 4;
            samples += 1;
        }
        assert_eq!(pos, bytes.len(), "truncated trailing sample");
        (samples, games)
    }

    // Several game threads over one sink and one batching net: the file is still a
    // sequence of whole games, every thread's games are in it, and the gamelog has
    // one line per decided game.
    #[test]
    fn threads_share_sink_and_batch_server() {
        let dir = std::env::temp_dir().join(format!("taflzero-threads-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("w0.bin");
        let log = dir.join("w0.gamelog");
        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_file(&log);

        let threads = 3;
        let server = crate::search::nn::BatchServer::start(
            "copenhagen11x11.onnx",
            11,
            threads * 8,
            std::time::Duration::from_millis(2),
            false,
        )
        .unwrap();
        let sink = DatagenSink::open(
            out.to_str().unwrap(),
            log.to_str().unwrap(),
            RulesEnum::Copenhagen11x11,
        );
        let cfg = DatagenConfig {
            search: SearchConfig {
                full_nodes: 2,
                cheap_nodes: 2,
                full_prob: 1.0,
                gumbel: None,
                // One leaf per net call: the batch is pooled across the threads.
                search_batch: 1,
                ..SearchConfig::default()
            },
            limits: GameLimits {
                max_plies: 6,
                max_no_capture: 400,
            },
            ..DatagenConfig::default()
        };
        std::thread::scope(|scope| {
            for t in 0..threads {
                let server = &server;
                let sink = &sink;
                let mut cfg = cfg.clone();
                cfg.daemon.thread = Some(t);
                scope.spawn(move || {
                    let mut nn = NeuralNet::shared(std::sync::Arc::clone(server));
                    server.client_started();
                    // Two games' worth of positions per thread (6 plies each).
                    gen_train_data_into(sink, &mut nn, Some(12), RulesEnum::Copenhagen11x11, cfg);
                    server.client_stopped();
                });
            }
        });
        drop(sink);

        let bytes = std::fs::read(&out).unwrap();
        let (samples, games) = count_samples(&bytes);
        assert!(samples >= threads * 12, "{samples} samples");
        assert!(games >= threads * 2, "{games} games");
        let (batches, served) = server.stats();
        assert!(served >= samples as usize);
        assert!(batches > 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Self-play writes workerN.feat next to workerN.bin; replaying the .bin with the
    // `taflzero --features` code gives the same bytes. A side file a game short (the
    // worker died between its two writes) is completed when the files are reopened.
    #[test]
    fn datagen_side_file_matches_the_replay_byte_for_byte() {
        let dir = std::env::temp_dir().join(format!("taflzero-feat-dg-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("worker0.bin");
        let log = dir.join("worker0.gamelog");
        let mut nn = NeuralNet::new("copenhagen11x11.onnx", 11, 8).unwrap();
        let cfg = DatagenConfig {
            search: SearchConfig {
                full_nodes: 3,
                cheap_nodes: 2,
                full_prob: 0.5,
                gumbel: None,
                ..SearchConfig::default()
            },
            limits: GameLimits {
                max_plies: 160,
                max_no_capture: 400,
            },
            ..DatagenConfig::default()
        };
        gen_train_data(
            out.to_str().unwrap(),
            log.to_str().unwrap(),
            &mut nn,
            Some(400),
            RulesEnum::Copenhagen11x11,
            cfg,
        );
        let feat = std::fs::read(feature_path(out.to_str().unwrap())).unwrap();
        let replayed = dir.join("replayed.feat");
        let stats = crate::search::feature_dump::dump_features(
            &out,
            &replayed,
            None,
            RulesEnum::Copenhagen11x11,
            2,
        )
        .unwrap();
        println!("{}", stats.summary());
        assert!(stats.records >= 400);
        assert_eq!(
            (feat.len() - 16) / 144,
            stats.records,
            "one record per sample"
        );
        let want = std::fs::read(&replayed).unwrap();
        // Surprise and FLAGS2 are self-play only; the replay writes 0 there.
        let mut feat = feat;
        for r in 0..stats.records {
            let at = 16 + r * 144;
            feat[at + crate::search::features::offset::SURPRISE] = 0;
            feat[at + crate::search::features::offset::FLAGS2] = 0;
        }
        if feat != want {
            let at = feat.iter().zip(&want).position(|(a, b)| a != b).unwrap();
            let r = (at - 16) / 144;
            let rec = |b: &[u8]| b[16 + r * 144..16 + (r + 1) * 144].to_vec();
            panic!(
                "side files differ at byte {at} (record {r}, offset {}): datagen {} replay {}
 datagen record {:?}
 replay  record {:?}",
                (at - 16) % 144,
                feat[at],
                want[at],
                &rec(&feat)[72..80],
                &rec(&want)[72..80]
            );
        }

        // Drop the last game's records and reopen: they are replayed back in.
        let bytes = std::fs::read(&out).unwrap();
        let (samples, games) = count_samples(&bytes);
        assert!(games >= 2);
        let last_game = {
            let mask = crate::mcts::export::legal_mask_bytes(11);
            let mut pos = 0usize;
            let mut starts = vec![0usize];
            let mut i = 0;
            while pos < bytes.len() {
                let len_at = pos + 50 + mask;
                let plen = u16::from_le_bytes([bytes[len_at], bytes[len_at + 1]]) as usize;
                let flags = bytes[len_at + 2 + plen * 4 + 2];
                pos = len_at + 2 + plen * 4 + 4;
                i += 1;
                if flags & FLAG_LAST_OF_GAME != 0 {
                    starts.push(i);
                }
            }
            samples - starts[starts.len() - 2]
        };
        let feat_path = feature_path(out.to_str().unwrap());
        std::fs::write(&feat_path, &feat[..feat.len() - last_game * 144]).unwrap();
        drop(DatagenSink::open(
            out.to_str().unwrap(),
            log.to_str().unwrap(),
            RulesEnum::Copenhagen11x11,
        ));
        assert_eq!(std::fs::read(&feat_path).unwrap(), want);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn worker_overrides_parse_key_values_and_reject_junk() {
        let o = WorkerOverrides::parse(
            "# a comment
threads = 2048
search_batch=1 # one leaf
ramp_seconds=600

",
        )
        .unwrap();
        assert_eq!(
            o,
            WorkerOverrides {
                threads: Some(2048),
                search_batch: Some(1),
                ramp_seconds: Some(600)
            }
        );
        assert_eq!(
            WorkerOverrides::parse("").unwrap(),
            WorkerOverrides::default()
        );
        assert!(WorkerOverrides::parse("threads: 4").is_err());
        assert!(WorkerOverrides::parse("threads=four").is_err());
        assert!(WorkerOverrides::parse("nodes=4").is_err());
        // Zero would stop a worker dead; it is read as one.
        assert_eq!(
            WorkerOverrides::parse("threads=0").unwrap().threads,
            Some(1)
        );
    }

    #[test]
    fn worker_overrides_are_read_next_to_the_data_file() {
        let dir = std::env::temp_dir().join(format!("taflzero-ovr-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let data = dir.join("worker0.bin");
        assert!(WorkerOverrides::next_to(data.to_str().unwrap()).is_none());
        std::fs::write(
            dir.join(WorkerOverrides::FILE),
            "search_batch=1
",
        )
        .unwrap();
        let (file, parsed) = WorkerOverrides::next_to(data.to_str().unwrap()).unwrap();
        assert!(file.ends_with(WorkerOverrides::FILE));
        assert_eq!(parsed.unwrap().search_batch, Some(1));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
