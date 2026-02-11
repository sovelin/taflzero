use std::fs::File;
use crate::Board;
use crate::board::constants::INITIAL_FEN;
use crate::mcts::export::PendingSample;
use crate::mcts::mcts::{mcts_search, MCTSTree};
use crate::search::nn::NeuralNet;
use crate::search_data::SearchData;
use crate::terminal::check_terminal;

fn play_game(nn: &mut NeuralNet, search_data: &mut SearchData) -> Vec<PendingSample> {
    let mut res = vec![];
    let mut board: Board = Board::new();
    board.set_fen(INITIAL_FEN).expect("Invalid FEN");

    let game_result;

    loop {
        let mut mcts_tree = MCTSTree::new();
        let mv = mcts_search(&mut board, &mut mcts_tree, nn, search_data, None);

        if let Some(mv) = mv {
            res.push(mcts_tree.make_pending_sample(&board));
            board.make_move_simple(mv).expect("Make move failed");

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

pub fn gen_train_data(output_path: &str, num_games: usize) {
    let mut nn = NeuralNet::new("./random_init.onnx");
    let mut search_data = SearchData::new();

    let file = File::create(output_path).expect("Failed to create output file");

    loop {
        let res = play_game(&mut nn, &mut search_data);

    }
}