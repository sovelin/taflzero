use std::fs::{OpenOptions};
use std::io::BufWriter;
use rand::prelude::StdRng;
use rand::Rng;
use crate::{Board, PRECOMPUTED};
use crate::mcts::export::PendingSample;
use crate::mcts::mcts::{mcts_search, MCTSConfig, MCTSTree};
use crate::movegen::MoveGen;
use crate::search::nn::NeuralNet;
use crate::search_data::SearchData;
use crate::terminal::check_terminal;
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


fn play_game(nn: &mut NeuralNet, search_data: &mut SearchData) -> Vec<PendingSample> {
    let mut board = set_random_position(&mut search_data.random_generator);
    let mut res = vec![];

    let mut config = MCTSConfig::default_train();
    let game_result;
    let mut move_number: usize = 0;
    let mut mcts_tree = MCTSTree::new();

    loop {
        config.temperature = if move_number < 20 { 1.0 } else { 0.0 };
        let mv = mcts_search(&mut board, &mut mcts_tree, nn, search_data, None, Some(400), &config);
        move_number += 1;

        if let Some(mv) = mv {
            res.push(mcts_tree.make_pending_sample(&board));
            board.make_move_simple(mv).expect("Make move failed");
            mcts_tree.reroot(mv);

            if let Some(result) = check_terminal(&mut board) {
                game_result = Some(result);
                break;
            }
        } else {
            game_result = None;
            break;
        }
    }

    for sample in res.iter_mut() {
        sample.set_value_from_result(game_result);
    }

    // print game result
    match game_result {
        Some(terminal_result) => {
            match terminal_result {
                Side::ATTACKERS => println!("Attacker wins"),
                Side::DEFENDERS => println!("Defender wins"),
            }
        },
        None => println!("Game ended without terminal result"),
    }

    res
}

pub fn gen_train_data(output_path: &str, nn: &mut NeuralNet, game_limit: Option<usize>) {
    let mut search_data = SearchData::new();

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)
        .expect("Could not open output file");

    let mut writer = BufWriter::new(file);
    let mut games_generated = 0usize;

    loop {
        if let Some(limit) = game_limit {
            if games_generated >= limit {
                println!("Datagen finished: generated {} games", games_generated);
                break;
            }
        }

        let res = play_game(nn, &mut search_data);
        println!("Generated a game with {} samples", res.len());

        for sample in res {
            sample.write_to(&mut writer).expect("Cannot write sample");
        }

        games_generated += 1;

    }
}
