pub mod gen_train_data;
pub mod mcts;
pub mod nn;
pub mod search_data;
pub mod search_root;

use crate::search::nn::NUM_PLANES;

pub fn get_sample_size(board_size: usize) -> usize {
    NUM_PLANES * board_size * board_size
}
