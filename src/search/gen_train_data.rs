use std::fs::{OpenOptions};
use std::io::{BufWriter, Write};
use rand::prelude::StdRng;
use rand::Rng;
use crate::{Board, PRECOMPUTED};
use crate::movegen::MoveGen;
use crate::mcts::export::{LegalMask, PendingSample};
use crate::mcts::utils::move_to_policy_index;
use crate::position_export::BitPosition;
use crate::mcts::mcts::{mcts_search, MCTSConfig, MCTSTree};
use crate::search::nn::NeuralNet;
use crate::search_data::SearchData;
use crate::terminal::{check_terminal, get_terminal, is_threefold_repetition, TerminalType};
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


fn terminal_type_str(t: &TerminalType) -> &'static str {
    match t {
        TerminalType::KingCaptured => "atk_capture",
        TerminalType::ThreefoldRepetition => "atk_threefold",
        TerminalType::KingOnCorner => "def_corner",
        TerminalType::DefendersSurrounded => "atk_surrounded",
        TerminalType::FortCheck => "def_fort",
    }
}

fn play_game(nn: &mut NeuralNet, search_data: &mut SearchData) -> (Vec<PendingSample>, Option<Side>, Option<&'static str>) {
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
    let mut terminal_str: Option<&'static str> = None;
    let mut move_number: usize = 0;
    let mut mcts_tree = MCTSTree::new();
    let mut no_capture_counter = 0;

    loop {
        config.temperature = if move_number < 60 { 1.0 } else { 0.0 };
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

            if no_capture_counter >= 500 || move_number >= 700 {
                // end the game as a draw
                game_result = None;
                terminal_str = Some(if no_capture_counter >= 500 { "draw_nocapture" } else { "draw_limit" });
                break;
            }

            // Treat threefold repetition as draw for training
            // if is_threefold_repetition(&board) {
            //     game_result = None;
            //     break;
            // }

            if let Some(terminal) = get_terminal(&mut board) {
                let result = check_terminal(&mut board).unwrap();
                // threefold repetition can also cause terminal, but we want to treat it as draw for training
                if result == Side::ATTACKERS && is_threefold_repetition(&board) {
                    game_result = None;
                    terminal_str = Some("draw_threefold");
                } else {
                    println!("{}", board);
                    terminal_str = Some(terminal_type_str(&terminal));
                    game_result = Some(result);
                }

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

    (res, game_result, terminal_str)
}

pub fn gen_train_data(output_path: &str, log_path: &str, nn: &mut NeuralNet, game_limit: Option<usize>) {
    let mut search_data = SearchData::new();

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

        let (res, game_result, terminal_str) = play_game(nn, &mut search_data);
        if game_result.is_none() { continue; }


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

        // Write to gamelog: terminal_type,game_length
        if let Some(t) = terminal_str {
            writeln!(log_writer, "{},{}", t, res.len()).expect("Cannot write gamelog");
            log_writer.flush().expect("Cannot flush gamelog");
        }

        for sample in res {
            sample.write_to(&mut writer).expect("Cannot write sample");
        }

    }
}

pub fn dump_single_sample(output_path: &str) {
    let mut board = Board::new();
    board.setup_initial_position().expect("Setup initial position failed");

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
    let sample = PendingSample::from_manual(
        BitPosition::from_board(&board),
        legal_mask,
        policy,
        1,
    );

    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(output_path)
        .expect("Could not open output file");
    let mut writer = BufWriter::new(file);
    sample.write_to(&mut writer).expect("Cannot write sample");

    println!("DUMP_SAMPLE index={} legal_moves={}", move_index, move_gen.count);
}
