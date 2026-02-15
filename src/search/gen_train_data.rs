use std::fs::{OpenOptions};
use std::io::BufWriter;
use rand::prelude::StdRng;
use rand::Rng;
use crate::{Board, PRECOMPUTED};
use crate::mcts::export::PendingSample;
use crate::mcts::mcts::{mcts_search, MCTSConfig, MCTSTree};
use crate::search::nn::NeuralNet;
use crate::search_data::SearchData;
use crate::terminal::{check_terminal, is_threefold_repetition};
use crate::types::{Piece, Side};

fn set_piece_to_random_square(
    board: &mut Board,
    empty_squares: &mut Vec<usize>,
    rnd: &mut StdRng,
    piece: Piece,
) {
    if empty_squares.is_empty() {
        return;
    }

    let idx = rnd.gen_range(0..empty_squares.len());
    let sq = empty_squares.swap_remove(idx);

    board.set_piece(sq, piece).expect("set_piece");
}

fn set_random_position(rnd: &mut StdRng) -> Board {
    let mut board = Board::new();
    let mut empty_squares: Vec<usize> = (0..board.board.len())
        .filter(|&sq| board.board[sq] == Piece::EMPTY)
        .filter(|&sq| !PRECOMPUTED.corners_sq.contains(&sq))
        .collect();

    let attacker_pieces_count = rnd.gen_range(24..=50);
    let defender_pieces_count = rnd.gen_range(0..=12);

    set_piece_to_random_square(
        &mut board,
        &mut empty_squares,
        rnd,
        Piece::KING,
    );

    for _ in 0..attacker_pieces_count {
        set_piece_to_random_square(
            &mut board,
            &mut empty_squares,
            rnd,
            Piece::ATTACKER,
        );
    }

    for _ in 0..defender_pieces_count {
        set_piece_to_random_square(
            &mut board,
            &mut empty_squares,
            rnd,
            Piece::DEFENDER,
        );
    }

    board
}


fn play_game(nn: &mut NeuralNet, search_data: &mut SearchData) -> (Vec<PendingSample>, Option<Side>) {
    // let mut board = set_random_position(&mut search_data.random_generator);
    let mut board = Board::new();
    board.setup_initial_position().expect("Setup initial position failed");

    // Aggressive setup for start
    // board.set_fen("1aaaaaaaaa1/3aaaaa3/11/aa3d3aa/a3ddd3a/aa1ddkdd1aa/a3ddd3a/aa3d3aa/11/3aaaaa3/1aaaaaaaaa1 a").expect("Set fen failed");

    //board.set_fen("2aaaaaaa2/4aaa4/11/aa3d3aa/a3ddd3a/aa1ddkdd1aa/a3ddd3a/aa3d3aa/11/3aaaaa3/2aaaaaaa2 a").expect("Set fen failed");
   // board.set_fen("2aaaaaaa2/5a5/11/aa3d3aa/a3ddd3a/aa1ddkdd1aa/a3ddd3a/aa3d3aa/11/4aaa4/2aaaaaaa2 a").expect("Set fen failed");


    let mut res = vec![];

    let mut config = MCTSConfig::default_train();
    let game_result;
    let mut move_number: usize = 0;
    let mut mcts_tree = MCTSTree::new();
    let mut no_capture_counter = 0;

    loop {
        config.temperature = if move_number < 30 { 1.0 } else { 0.0 };
        // if board.side_to_move == Side::DEFENDERS {
        //     mcts_tree = MCTSTree::new();
        // }
        let iterations = if board.side_to_move == Side::ATTACKERS {
            400
        } else {
            400
        };

        let mv = mcts_search(&mut board, &mut mcts_tree, nn, search_data, None, Some(iterations), &config);
        move_number += 1;

        if let Some(mv) = mv {
            res.push(mcts_tree.make_pending_sample(&board));
            let pieces_count = board.attackers_count + board.defenders_count;
            board.make_move_simple(mv).expect("Make move failed");
            let new_pieces_count = board.attackers_count + board.defenders_count;

            if new_pieces_count != pieces_count {
                // reset no capture counter
                no_capture_counter = 0;
            } else {
                no_capture_counter += 1;
            }

            if no_capture_counter >= 200 || move_number >= 300 {
                // end the game as a draw
                game_result = None;
                break;
            }

            // Treat threefold repetition as draw for training
            if is_threefold_repetition(&board) {
                game_result = None;
                break;
            }

            if let Some(result) = check_terminal(&mut board) {
                game_result = Some(result);
                break;
            }
        } else {
            game_result = if board.side_to_move == Side::ATTACKERS {
                Some(Side::DEFENDERS)
            } else {
                // print board
                println!("Board:\n{}", board);

                Some(Side::ATTACKERS)
            };
            break;
        }
    }

    for sample in res.iter_mut() {
        sample.set_value_from_result(game_result);
    }

    (res, game_result)
}

pub fn gen_train_data(output_path: &str, nn: &mut NeuralNet, game_limit: Option<usize>) {
    let mut search_data = SearchData::new();

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)
        .expect("Could not open output file");

    let mut writer = BufWriter::new(file);
    let mut positions_generated = 0usize;
    let mut games_saved = 0usize;
    let mut attacker_wins_saved = 0usize;
    let mut defender_wins_saved = 0usize;
    let mut draws_saved = 0usize;
    let mut defender_wins_skipped = 0usize;
    const DEFENDER_WIN_KEEP_EVERY: usize = 1; // 1 = keep all, N>1 = keep 1 out of every N defender wins

    loop {
        if let Some(limit) = game_limit {
            if positions_generated >= limit {
                println!("Datagen finished: generated {} positions", positions_generated);
                break;
            }
        }

        let (res, game_result) = play_game(nn, &mut search_data);
        let is_defender_win = game_result == Some(Side::DEFENDERS);

        if is_defender_win {
            defender_wins_skipped += 1;
            if defender_wins_skipped % DEFENDER_WIN_KEEP_EVERY != 0 {
                continue;
            }
        }

        positions_generated += res.len();
        games_saved += 1;
        match game_result {
            Some(Side::ATTACKERS) => attacker_wins_saved += 1,
            Some(Side::DEFENDERS) => defender_wins_saved += 1,
            None => draws_saved += 1,
        }

        let total_saved = attacker_wins_saved + defender_wins_saved + draws_saved;
        let atk_pct = if total_saved > 0 { attacker_wins_saved as f64 / total_saved as f64 * 100.0 } else { 0.0 };
        let avg_game_len = if games_saved > 0 { positions_generated as f64 / games_saved as f64 } else { 0.0 };
        let result_str = match game_result {
            Some(Side::ATTACKERS) => "ATK WIN",
            Some(Side::DEFENDERS) => "DEF WIN",
            None => "DRAW",
        };
        println!("{} | game #{} ({} samples) | atk={} def={} draw={} | atk%={:.1}% | avg_len={:.1} | positions={}",
            result_str, games_saved, res.len(), attacker_wins_saved, defender_wins_saved, draws_saved, atk_pct, avg_game_len, positions_generated);

        for sample in res {
            sample.write_to(&mut writer).expect("Cannot write sample");
        }

    }
}
