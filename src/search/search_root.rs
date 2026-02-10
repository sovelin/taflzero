use std::cmp::{max, min};
use wasm_bindgen::prelude::wasm_bindgen;
use crate::board::Board;
use crate::evaluation::MATE_SCORE;
use crate::is_mate_score;
use crate::mcts::mcts::mcts_search;
use crate::moves::mv::Move;
use crate::search::constants::MAX_PLY;
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
    mcts_search(board, search_data, on_iteration);

    let mut best_score = 0;

    search_data.timer.start();
    search_data.nodes_searched = 0;
    search_data.best_move = None;

    let mut local_best_move: Option<Move> = None;

    for i in 1..=search_data.depth_limit {
        best_score = aspiration_window(board, search_data, tt, i, best_score);

        if search_data.time_exceeded() {
            break;
        }

        let time_elapsed = search_data.timer.elapsed_ms();
        local_best_move = search_data.best_move;

        if let Some(callback) = on_iteration {
            let best_move = search_data.best_move.unwrap_or_default();

            let speed = if time_elapsed > 0 {
                (search_data.nodes_searched * 1000) / time_elapsed
            } else {
                0
            };

            callback(SearchIterationResponse {
                depth: i as i32,
                mv: best_move,
                score: best_score,
                nodes: search_data.nodes_searched,
                time: time_elapsed,
                speed,
            });
        }
    }

    search_data.tt_age += 1;

    SearchResponse {
        best_move: local_best_move.unwrap_or_default(),
        score: best_score,
    }
}