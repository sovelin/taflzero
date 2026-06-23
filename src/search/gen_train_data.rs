use crate::board::position_export::BitPosition;
use crate::board::rules::RulesEnum;
use crate::board::types::{Piece, Side};
use crate::board::{Board, PRECOMPUTED};
use crate::mcts::export::{LegalMask, PendingSample};
use crate::mcts::mcts::{MCTSConfig, MCTSTree, mcts_search};
use crate::mcts::utils::move_to_policy_index;
use crate::movegen::MoveGen;
use crate::search::nn::NeuralNet;
use crate::search_data::SearchData;
use crate::terminal::{TerminalType, check_terminal, get_terminal};
use rand::RngExt;
use rand::SeedableRng;
use rand::prelude::StdRng;
use std::hash::{BuildHasher, Hasher};
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::{BufWriter, Read, Write};

const NODES_PER_MOVE: u64 = 100;
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
        let Ok(mut f) = std::fs::File::open(path) else { return };
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
            let bytes =
                unsafe { std::slice::from_raw_parts(bp as *const BitPosition as *const u8, bp_size) };
            let _ = f.write_all(bytes);
        }
    }
}

// ─── Position generation helpers ─────────────────────────────────────────────

fn set_piece_to_random_square(
    board: &mut Board,
    empty_squares: &mut Vec<usize>,
    rnd: &mut StdRng,
    piece: Piece,
) {
    if empty_squares.is_empty() {
        return;
    }
    let idx = rnd.random_range(0..empty_squares.len());
    let sq = empty_squares.swap_remove(idx);
    board.set_piece(sq, piece).expect("set_piece");
}

fn set_random_position(rnd: &mut StdRng) -> Board {
    let mut board = Board::new();
    let mut empty_squares: Vec<usize> = (0..board.board.len())
        .filter(|&sq| board.board[sq] == Piece::EMPTY)
        .filter(|&sq| !PRECOMPUTED.corners_sq.contains(&sq))
        .collect();

    let attacker_pieces_count = rnd.random_range(24..=50);
    let defender_pieces_count = rnd.random_range(0..=12);

    set_piece_to_random_square(&mut board, &mut empty_squares, rnd, Piece::KING);

    for _ in 0..attacker_pieces_count {
        set_piece_to_random_square(&mut board, &mut empty_squares, rnd, Piece::ATTACKER);
    }

    for _ in 0..defender_pieces_count {
        set_piece_to_random_square(&mut board, &mut empty_squares, rnd, Piece::DEFENDER);
    }

    board
}

// ─── Terminal type string ─────────────────────────────────────────────────────

fn terminal_type_str(t: &TerminalType) -> &'static str {
    match t {
        TerminalType::KingCaptured => "atk_capture",
        TerminalType::ThreefoldRepetition => "atk_threefold",
        TerminalType::KingOnCorner => "def_corner",
        TerminalType::DefendersSurrounded => "atk_surrounded",
        TerminalType::FortCheck => "def_fort",
    }
}

// ─── Game loop ────────────────────────────────────────────────────────────────

fn play_game(
    nn: &mut NeuralNet,
    search_data: &mut SearchData,
    variant: RulesEnum,
    start_board: Board,
) -> (Vec<PendingSample>, Vec<BitPosition>, Option<Side>, Option<&'static str>) {
    let mut board = start_board;

    let mut res: Vec<PendingSample> = vec![];
    let mut board_history: Vec<BitPosition> = vec![];

    let mut config = MCTSConfig::default_train();
    let game_result;
    let mut terminal_str: Option<&'static str> = None;
    let mut move_number: usize = 0;
    let mut mcts_tree = MCTSTree::new();
    let mut no_capture_counter = 0;

    loop {
        config.temperature = if move_number < 60 { 1.0 } else { 0.0 };

        let mv = mcts_search(
            &mut board,
            &mut mcts_tree,
            nn,
            search_data,
            None,
            Some(NODES_PER_MOVE),
            &config,
            None,
        );
        move_number += 1;

        if let Some(mv) = mv {
            res.push(mcts_tree.make_pending_sample(&board));

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

            if no_capture_counter >= 500 || move_number >= 700 {
                game_result = None;
                terminal_str = Some(if no_capture_counter >= 500 {
                    "draw_nocapture"
                } else {
                    "draw_limit"
                });
                break;
            }

            if let Some(terminal) = get_terminal(&mut board) {
                let result = check_terminal(&mut board).unwrap();
                println!("{}", board);
                terminal_str = Some(terminal_type_str(&terminal));
                game_result = Some(result);
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

    for sample in res.iter_mut() {
        sample.set_value_from_result(game_result);
    }

    (res, board_history, game_result, terminal_str)
}

// ─── Main datagen loop ────────────────────────────────────────────────────────

pub struct DatagenConfig {
    pub curriculum_fraction: f64,
    pub curriculum_path: Option<String>,
    pub curriculum_max_size: usize,
}

impl Default for DatagenConfig {
    fn default() -> Self {
        Self {
            curriculum_fraction: 0.0,
            curriculum_path: None,
            curriculum_max_size: 50_000,
        }
    }
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

    if use_curriculum {
        if let Some(ref path) = cfg.curriculum_path {
            curriculum.load_from_file(path);
            curriculum.trim_and_rewrite_file(path);
        }
    }

    // Append file handle for curriculum (opened lazily)
    let mut curriculum_file: Option<std::fs::File> = None;
    if use_curriculum {
        if let Some(ref path) = cfg.curriculum_path {
            curriculum_file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .ok();
        }
    }

    let bp_size = size_of::<BitPosition>();

    let mut positions_generated = 0usize;
    let mut games_played = 0usize;
    let mut attacker_wins_saved = 0usize;
    let mut defender_wins_saved = 0usize;
    let mut draws_skipped = 0usize;
    let mut curriculum_games = 0usize;
    let mut curriculum_decisive = 0usize;

    loop {
        if let Some(limit) = game_limit
            && positions_generated >= limit
        {
            println!(
                "Datagen finished: generated {} positions",
                positions_generated
            );
            break;
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
            b.setup_initial_position().expect("Setup initial position failed");
            b
        };

        let (res, board_history, game_result, terminal_str) =
            play_game(nn, &mut search_data, variant, start_board);

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
            None => draws_skipped += 1,
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
            None => "DRAW",
        };
        let cur_tag = if is_curriculum_game { " [CUR]" } else { "" };
        println!(
            "{}{} | game #{} ({} samples) | atk={} def={} draw={} | atk%={:.1}% | avg_len={:.1} | positions={}{}",
            result_str,
            cur_tag,
            games_played,
            res.len(),
            attacker_wins_saved,
            defender_wins_saved,
            draws_skipped,
            atk_pct,
            avg_game_len,
            positions_generated,
            if curriculum_games > 0 {
                format!(" | cur_decisive={}/{}", curriculum_decisive, curriculum_games)
            } else {
                String::new()
            }
        );

        // Gamelog
        if let Some(t) = terminal_str {
            writeln!(log_writer, "{},{}", t, res.len()).expect("Cannot write gamelog");
            log_writer.flush().expect("Cannot flush gamelog");
        }

        // Only write decisive games to training data
        if game_result.is_some() {
            positions_generated += res.len();
            for sample in &res {
                sample.write_to(&mut writer).expect("Cannot write sample");
            }

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
    let move_index = move_to_policy_index(first_mv);

    let mut legal_mask = LegalMask::new();
    for i in 0..move_gen.count {
        let mv = move_gen.moves[i];
        let idx = move_to_policy_index(mv);
        legal_mask.set(idx as usize);
    }

    let policy = vec![(move_index, 7)];
    let sample =
        PendingSample::from_manual(BitPosition::from_board(&board, 1), legal_mask, policy, 1);

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
