use wasm_bindgen::prelude::wasm_bindgen;
use crate::board::Board;
use crate::mcts::mcts::{mcts_search, MCTSConfig, MCTSTree};
use crate::moves::mv::Move;
use crate::search::nn::NeuralNet;
use crate::search::search_data::SearchData;
use crate::search::transposition::TranspositionTable;

#[wasm_bindgen]
pub struct SearchIterationResponse {
    pub score: i32,
    pub nodes: u64,
    pub time: u64,
    pub speed: u64,
    pub(crate) pv: Vec<Move>,
    pub winrate: f32,
    pub multi_pv: Option<usize>,
}

impl SearchIterationResponse {
    pub fn pv(&self) -> &[Move] {
        &self.pv
    }
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
    nn: &mut NeuralNet,
    on_iteration: Option<&dyn Fn(SearchIterationResponse)>,
    tree: &mut MCTSTree,
    multi_pv: Option<usize>
) -> SearchResponse {

    let config = MCTSConfig::default_play();
    let best_move = mcts_search(board, tree, nn, search_data, on_iteration, None, &config, multi_pv);

    SearchResponse {
        best_move: best_move.unwrap_or_default(),
        score: 0,
    }
}

pub fn search_root_nodes(
    board: &mut Board,
    search_data: &mut SearchData,
    tt: &mut TranspositionTable,
    nn: &mut NeuralNet,
    on_iteration: Option<&dyn Fn(SearchIterationResponse)>,
    tree: &mut MCTSTree,
    nodes: u64,
    multi_pv: Option<usize>
) -> SearchResponse {

    let config = MCTSConfig::default_play();
    let best_move = mcts_search(board, tree, nn, search_data, on_iteration, Some(nodes), &config, multi_pv);

    SearchResponse {
        best_move: best_move.unwrap_or_default(),
        score: 0,
    }
}
