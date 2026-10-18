use crate::board::Board;
use crate::board::position_export::BitPosition;
use crate::board::rules::RulesEnum;
use crate::board::types::Side;
use crate::mcts::export::{KING_CORNER_NONE, LegalMask, PendingSample, king_corner_index};
use crate::mcts::utils::move_to_policy_index;
use crate::mcts::{C_PUCT, MCTSConfig, MCTSTree, mcts_search};
use crate::movegen::MoveGen;
use crate::search::nn::NeuralNet;
use crate::search_data::SearchData;
use crate::terminal::{TerminalType, check_terminal, get_terminal};
use rand::RngExt;
use rand::SeedableRng;
use rand::prelude::StdRng;
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::hash::{BuildHasher, Hasher};
use std::io::{BufWriter, Read, Write};

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

fn terminal_type_str(t: &TerminalType) -> &'static str {
    match t {
        TerminalType::KingCaptured => "atk_capture",
        TerminalType::ThreefoldRepetition => "atk_threefold",
        TerminalType::KingOnCorner => "def_corner",
        TerminalType::DefendersSurrounded => "atk_surrounded",
        TerminalType::FortCheck => "def_fort",
        TerminalType::KingOnEdge => "def_edge",
    }
}

// ─── Game loop ────────────────────────────────────────────────────────────────

fn play_game(
    nn: &mut NeuralNet,
    search_data: &mut SearchData,
    start_board: Board,
    search_cfg: &SearchConfig,
    limits: &GameLimits,
    rng: &mut StdRng,
) -> (
    Vec<PendingSample>,
    Vec<BitPosition>,
    Option<Side>,
    Option<&'static str>,
) {
    let mut board = start_board;

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
        let (config, nodes) = if is_full {
            (&full_config, search_cfg.full_nodes)
        } else {
            (&cheap_config, search_cfg.cheap_nodes)
        };

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

            let rep = board.rep_table.get(&board.zobrist).copied().unwrap_or(1);
            board_history.push(BitPosition::from_board(&board, rep));

            let pieces_count = board.attackers_count + board.defenders_count;
            board.make_move_simple(mv).expect("Make move failed");
            let new_pieces_count = board.attackers_count + board.defenders_count;

            if new_pieces_count != pieces_count {
                no_capture_counter = 0;
            } else {
                no_capture_counter += 1;
            }

            // Terminals first: a game that ends exactly on the limit is still decided.
            if let Some(terminal) = get_terminal(&mut board) {
                println!("{}", board);
                terminal_str = Some(terminal_type_str(&terminal));
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

    (res, board_history, game_result, terminal_str)
}

// ─── Main datagen loop ────────────────────────────────────────────────────────

/// Per-move search budget (playout cap randomization).
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
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            full_nodes: 400,
            cheap_nodes: 100,
            full_prob: 1.0,
            gumbel: None,
        }
    }
}

/// When a game is cut off as undecided.
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
#[derive(Default)]
pub struct DaemonConfig {
    /// Reload the net before a game when the file's mtime changed.
    pub net_watch: bool,
    /// Finish the current game and exit when this file exists.
    pub stop_file: Option<String>,
    /// Do not start a new game while this file exists.
    pub pause_file: Option<String>,
    /// Log prefix only.
    pub worker_id: Option<usize>,
}

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
    let mut search_data = SearchData::new();
    let mut rng = StdRng::seed_from_u64(std::hash::RandomState::new().build_hasher().finish());

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)
        .expect("Could not open output file");
    let mut writer = BufWriter::new(file);

    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .expect("Could not open gamelog file");
    let mut log_writer = BufWriter::new(log_file);

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
    let tag = cfg
        .daemon
        .worker_id
        .map(|id| format!("[W{id}] "))
        .unwrap_or_default();
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

        let (res, board_history, game_result, terminal_str) = play_game(
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
            writer.write_all(&game_bytes).expect("Cannot write game");
            writer.flush().expect("Cannot flush game");
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
        };
        let (res, _, result, term) =
            play_game(&mut nn, &mut search_data, board, &search, &limits, &mut rng);
        (res, result, term)
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
        assert_eq!(
            terminal_type_str(&TerminalType::ThreefoldRepetition),
            "atk_threefold"
        );
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
}
