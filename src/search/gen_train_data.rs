use std::fs::{OpenOptions};
use std::io::BufWriter;
use crate::Board;
use crate::board::constants::INITIAL_FEN;
use crate::mcts::export::PendingSample;
use crate::mcts::mcts::{mcts_search, MCTSConfig, MCTSTree};
use crate::search::nn::NeuralNet;
use crate::search_data::SearchData;
use crate::terminal::check_terminal;

fn play_game(nn: &mut NeuralNet, search_data: &mut SearchData) -> Vec<PendingSample> {
    let mut res = vec![];
    let mut board: Board = Board::new();
    board.set_fen(INITIAL_FEN).expect("Invalid FEN");

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

    res
}

pub fn gen_train_data(output_path: &String) {
    let mut nn = NeuralNet::new("./random_init.onnx");
    let mut search_data = SearchData::new();

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)
        .expect("Could not open output file");

    let mut writer = BufWriter::new(file);

    loop {
        let res = play_game(&mut nn, &mut search_data);
        println!("Generated a game with {} samples", res.len());

        for sample in res {
            sample.write_to(&mut writer).expect("Cannot write sample");
        }

    }
}