use wasm_bindgen::prelude::wasm_bindgen;
use crate::board::Board;
use crate::evaluation::MATE_SCORE;
use crate::moves::mv::Move;
use crate::search::constants::MAX_PLY;
use crate::search::search::search;
use crate::search::search_data::SearchData;
use crate::search::transposition::TranspositionTable;

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

pub fn search_root(
    board: &mut Board,
    search_data: &mut SearchData,
    tt: &mut TranspositionTable,
    on_iteration: Option<&dyn Fn(SearchIterationResponse)>,
) -> SearchResponse {
    let best_score = 0;

    search_data.timer.start();

    for i in 1..MAX_PLY {
        let res = search(board, i as u32, -MATE_SCORE, MATE_SCORE, 0, search_data, tt);

        if search_data.time_exceeded() {
            break;
        }

        let time_elapsed = search_data.timer.elapsed_ms();

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
                score: res,
                nodes: search_data.nodes_searched,
                time: time_elapsed,
                speed,
            });
        }
    }

    SearchResponse {
        best_move: search_data.best_move.unwrap_or_default(),
        score: best_score,
    }
}