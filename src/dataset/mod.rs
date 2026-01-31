use std::fs::OpenOptions;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use crate::{Board, Engine};
use crate::board::constants::INITIAL_FEN;
use crate::movegen::MoveGen;
use crate::nnue::{load_fc1_from_raw, load_fc2_from_raw, Weights1, Weights2};
use crate::terminal::is_threefold_repetition;
use crate::types::Side;

struct GameResult {
    winner: Side,
    moves_count: usize,
    aborted: bool,
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
    set_random_opening(&mut engine, rnd, 32);
    let to_remove = 0; // rnd.gen_range(0..12);

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
        if is_threefold_repetition(&mut engine.board()) {
            return GameResult {
                winner: Side::ATTACKERS,
                moves_count,
                aborted: true,
            };
        }

        if let Some(res) = engine.check_terminal() {
            return GameResult {
                winner: res,
                moves_count,
                aborted: false,
            };
        }

        let bm = engine.make_search(1000000000, 3, None);
        if !bm.best_move.is_null() {
            engine.make_move(bm.best_move).unwrap();
            game.add_position(&engine.board());
            moves_count += 1;
            continue;
        } else {
            return GameResult {
                winner: Side::opposite(engine.board().side_to_move),
                moves_count,
                aborted: false,
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
    let mut game_saved = 0;
    let mut batcher = Batcher::new(batch_size);


    for i in 0..num_games {
        let mut learning_game = LearningGame::new();
        let result = play_random_game(&mut rng, &mut learning_game, &w1, &w2);

        if result.aborted {
            continue;
        }

        learning_game.mark_winner(result.winner);
        batcher.add_game(learning_game);

        // if batcher.is_full() {
        //     game_saved += batch_size;
        //     println!("Saved {} positions to file...", game_saved);
        //     batcher.save_to_file(file_name.as_str());
        //     batcher.clear();
        // } else if i % 10 == 0 {
        //     batcher.print_fullness();
        // }
            game_saved += batch_size;
            println!("Saved {} positions to file...", game_saved);
            batcher.save_to_file(file_name.as_str());
            batcher.clear();

        total_moves += result.moves_count;

        match result.winner {
            Side::DEFENDERS => defender_wins += 1,
            Side::ATTACKERS => attacker_wins += 1,
        }
    }
}