use std::fs::OpenOptions;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use crate::{Board, Engine};
use crate::constants::INITIAL_FEN;
use crate::movegen::MoveGen;
use crate::nnue::{load_fc1_from_raw, load_fc1_single_line, load_fc2_from_raw, load_fc2_single_line, Weights1, Weights2};
use crate::types::Side;

struct GameResult {
    winner: Side,
    moves_count: usize,
}

#[derive(Clone)]
struct Fen {
    fen: String,
    stm: Side,
}

struct LearningGame {
    fens: Vec<Fen>,
    winner: Side,
    is_completed: bool,
}

impl LearningGame {
    pub fn new() -> Self {
        Self {
            fens: vec![],
            winner: Side::DEFENDERS,
            is_completed: false,
        }
    }

    pub fn add_position(&mut self, board: &Board) {
        if self.is_completed {
            return;
        }

        let fen = board.get_fen();
        self.fens.push(Fen {
            fen,
            stm: board.side_to_move,
        });
    }
    pub fn mark_winner(&mut self, winner: Side) {
        self.winner = winner;
        self.is_completed = true;
    }

    pub fn clear(&mut self) {
        self.fens.clear();
        self.is_completed = false;
    }

    pub fn fens(&self) -> &Vec<Fen> {
        &self.fens
    }
}

struct GameFen {
    fen: Fen,
    result: Side,
}

struct Batcher {
    positions: Vec<GameFen>,
    attackers_wins: usize,
    defenders_wins: usize,
    limit: usize,
    side_limit: usize,
}

impl Batcher {
    pub fn new(limit: usize) -> Self {
        let side_limit = limit / 2;

        Self {
            positions: Vec::new(),
            attackers_wins: 0,
            defenders_wins: 0,
            limit: side_limit * 2,
            side_limit
        }
    }

    pub fn add_game(&mut self, game: LearningGame) {
        if game.is_completed {
            for fen in game.fens() {
                if self.positions.len() >= self.limit {
                    break;
                }

                if game.winner == Side::DEFENDERS && self.defenders_wins < self.side_limit {
                    self.positions.push(GameFen {
                        fen: fen.clone(),
                        result: game.winner,
                    });
                    self.defenders_wins += 1;
                } else if game.winner == Side::ATTACKERS && self.attackers_wins < self.side_limit {
                    self.positions.push(GameFen {
                        fen: fen.clone(),
                        result: game.winner,
                    });
                    self.attackers_wins += 1;
                }
            }
        }
    }

    pub fn is_full(&self) -> bool {
        self.positions.len() >= self.limit
    }

    pub fn print_fullness(&self) {
        println!("Batcher fullness: {}/{}", self.positions.len(), self.limit);
        println!("  Attackers wins: {}/{}", self.attackers_wins, self.side_limit);
        println!("  Defenders wins: {}/{}", self.defenders_wins, self.side_limit);
    }

    pub fn save_to_file(&self, path: &str) {
        use std::io::Write;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();

        for fen in &self.positions {
            let win_score = if fen.result == fen.fen.stm { 1 } else { 0 };
            let line = format!("{},{}\n", fen.fen.fen, win_score);
            file.write_all(line.as_bytes()).unwrap();
        }

        file.flush().unwrap();
    }

    pub fn clear(&mut self) {
        self.positions.clear();
        self.attackers_wins = 0;
        self.defenders_wins = 0;
    }
}

static ATTACKER_OVERSAMPLE_FACTOR: usize = 5;


pub struct SaveResult {
    total_positions_written: usize,
    total_attacker_positions_written: usize,
    total_defender_positions_written: usize,
}

impl SaveResult {
    pub fn new() -> Self {
        Self {
            total_positions_written: 0,
            total_attacker_positions_written: 0,
            total_defender_positions_written: 0,
        }
    }

    pub fn append(&mut self, other: &SaveResult) {
        self.total_positions_written += other.total_positions_written;
        self.total_attacker_positions_written += other.total_attacker_positions_written;
        self.total_defender_positions_written += other.total_defender_positions_written;
    }
}

impl std::fmt::Display for SaveResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Total positions written: {}, Attacker positions written: {}, Defender positions written: {}, D/A ratio: {:.2}",
            self.total_positions_written,
            self.total_attacker_positions_written,
            self.total_defender_positions_written,
            if self.total_defender_positions_written == 0 {
                self.total_attacker_positions_written as f64
            } else {
                self.total_attacker_positions_written as f64 / self.total_defender_positions_written as f64
            }
        )
    }
}

impl std::fmt::Debug for SaveResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

pub fn save_to_file(lean: &LearningGame, file_path: &str, attackers_oversample: usize) -> SaveResult {
    use std::io::Write;
    let mut total_pos_written = 0;
    let mut total_attacker_pos_written = 0;
    let mut total_defender_pos_written = 0;

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(file_path)
        .unwrap();

    for fen in &lean.fens {
        let win_score = if lean.winner == fen.stm { 1 } else { 0 };
        let line = format!("{},{}\n", fen.fen, win_score);

        // min between 5 and attackers_oversample
        let attackers_oversample = if attackers_oversample < 5 {
            5
        } else {
            attackers_oversample as usize
        };

        if lean.winner == Side::ATTACKERS {
            total_attacker_pos_written += 1;
            for _ in 0..attackers_oversample {
                file.write_all(line.as_bytes()).unwrap();
                total_pos_written += 1;
            }
            continue;
        } else {
            file.write_all(line.as_bytes()).unwrap();
            total_pos_written += 1;
            total_defender_pos_written += 1;
        }
    }

    file.flush().unwrap();

    SaveResult {
        total_positions_written: total_pos_written,
        total_attacker_positions_written: total_attacker_pos_written,
        total_defender_positions_written: total_defender_pos_written,
    }
}


fn set_random_opening(engine: &mut Engine, rng: &mut StdRng, ply_count: usize) {
    let mut mv_generator = MoveGen::new();
    let mut moves = Vec::with_capacity(ply_count);

    let mut board = Board::new();
    board.setup_initial_position().unwrap();

    for _ in 0..ply_count {
        mv_generator.generate_moves(&board);

        if mv_generator.count == 0 {
            break;
        }

        let random_move = mv_generator.moves[rng.gen_range(0..mv_generator.count)];
        moves.push(random_move);
        board.make_move_simple(random_move).unwrap();
    }

    engine.set_position_and_moves(INITIAL_FEN, moves);
}

fn play_random_game(rnd: &mut StdRng, game: &mut LearningGame, w1: &Weights1, w2: &Weights2) -> GameResult {
    let mut engine = Engine::new(1, w1, w2);
    set_random_opening(&mut engine, rnd, 16);
    let to_remove = rnd.gen_range(0..12);

    if to_remove > 0 {
        for _ in 0..to_remove {
            let defenders_count = engine.get_board_mutable().defenders_count;

            if defenders_count == 0 {
                break;
            }

            let random_defender_index = rnd.gen_range(0..defenders_count);
            let sq = engine.get_board_mutable().defenders[random_defender_index as usize];
            engine.get_board_mutable()
                .clear_piece(sq);
        }
    }

    let mut moves_count: usize = 0;

    game.add_position(&engine.board());

    loop {
        if let Some(res) = engine.check_terminal() {
            return GameResult {
                winner: res,
                moves_count,
            };
        }

        let bm = engine.make_search(30, None);
        if !bm.best_move.is_null() {
            engine.make_move(bm.best_move).unwrap();
            game.add_position(&engine.board());
            moves_count += 1;
            continue;
        } else {
            // No best move found, return winner as opposite side
            if Side::opposite(engine.board().side_to_move) == Side::ATTACKERS {
                engine.print_board();
            }
            return GameResult {
                winner: Side::opposite(engine.board().side_to_move),
                moves_count,
            };
        }
    }
}

pub fn play_random_games(num_games: usize, file_name: String) {
    let time_seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let w1 = load_fc1_from_raw();
    let w2 = load_fc2_from_raw();

    let mut rng = StdRng::seed_from_u64(time_seed);

    let mut defender_wins = 0;
    let mut attacker_wins = 0;
    let mut total_moves = 0;

    let batch_size = 10000;

    let mut stat = SaveResult::new();



    for i in 0..num_games {
        let mut learning_game = LearningGame::new();
        let result = play_random_game(&mut rng, &mut learning_game, &w1, &w2);
        learning_game.mark_winner(result.winner);

        let over_sample_factor =
            if stat.total_attacker_positions_written == 0 {1} else {stat.total_defender_positions_written / (stat.total_attacker_positions_written)};

        let new_res = save_to_file(&learning_game, file_name.as_str(), over_sample_factor);
        stat.append(&new_res);
        println!("{:?}", stat);

        total_moves += result.moves_count;

        println!("Game {} completed. Winner: {:?}, Moves: {}, oversample: {}", i + 1, result.winner, result.moves_count, over_sample_factor);
    }
}