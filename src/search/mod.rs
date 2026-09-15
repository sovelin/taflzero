pub mod gen_train_data;
pub mod mcts;
pub mod nn;
pub mod search_data;
pub mod search_root;

use crate::search::nn::NUM_PLANES;

/// Move directions the policy head encodes: up, down, left, right.
pub const POLICY_DIRECTIONS: usize = 4;

pub fn get_sample_size(board_size: usize) -> usize {
    NUM_PLANES * board_size * board_size
}

/// One logit per (from-square, direction, distance) triple. A sliding piece can travel
/// 1..=board_size-1 squares, so the distance axis is one shorter than the board side.
pub fn get_policy_size(board_size: usize) -> usize {
    board_size * board_size * POLICY_DIRECTIONS * (board_size - 1)
}
