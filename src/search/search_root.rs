use std::cmp::{max, min};
use wasm_bindgen::prelude::wasm_bindgen;
use crate::board::Board;
use crate::evaluation::MATE_SCORE;
use crate::is_mate_score;
use crate::mcts::mcts::{mcts_search, MCTSConfig, MCTSTree};
use crate::moves::mv::Move;
use crate::search::constants::MAX_PLY;
use crate::search::nn::NeuralNet;
use crate::search::search::search;
use crate::search::search_data::SearchData;
use crate::search::transposition::TranspositionTable;
use crate::transposition::TTFlag;

#[wasm_bindgen]
pub struct SearchIterationResponse {
    pub depth: i32,
    pub mv: Move,
    pub score: i32,
    pub nodes: u64,
    pub time: u64,
    pub speed: u64,
}

#[wasm_bindgen]
pub struct SearchResponse {
    #[wasm_bindgen]
    pub best_move: Move,
    #[wasm_bindgen]
    pub score: i32,
}


fn aspiration_window(
    board: &mut Board,
    search_data: &mut SearchData,
    tt: &mut TranspositionTable,
    depth: u32,
    score: i32,
) -> i32 {
    let mut window_size = 50;

    let mut alpha = score - window_size;
    let mut beta = score + window_size;

    if depth < 3 || is_mate_score(score) {
        return search(board, depth, -MATE_SCORE, MATE_SCORE, 0, search_data, tt);
    }
    let mut f = score;

    while !is_mate_score(f) {
        f = search(
            board,
            depth,
            alpha,
            beta,
            0,
            search_data,
            tt,
        );

        let mut eval_type = TTFlag::Exact;

        if f > alpha && f <= beta {
            eval_type = TTFlag::Exact;
        }

        if f <= alpha {
            alpha = max(-MATE_SCORE, alpha - window_size);
            eval_type = TTFlag::UpperBound;
        } else if f >= beta {
            beta = min(MATE_SCORE, beta + window_size);
            window_size *= 2;
            eval_type = TTFlag::LowerBound;
        }

        window_size += window_size / 2;

        if eval_type == TTFlag::Exact {
            break;
        }
    }

    f
}

pub fn search_root(
    board: &mut Board,
    search_data: &mut SearchData,
    tt: &mut TranspositionTable,
    on_iteration: Option<&dyn Fn(SearchIterationResponse)>,
) -> SearchResponse {
    let mut tree = MCTSTree::new();
    let mut nn = NeuralNet::new("./gen1.onxx");

    let config = MCTSConfig::default_play();
    let best_move = mcts_search(board, &mut tree, &mut nn, search_data, on_iteration, None, &config);

    SearchResponse {
        best_move: best_move.unwrap_or_default(),
        score: 0,
    }
}