use wasm_bindgen::prelude::wasm_bindgen;
use crate::board::constants::{BOARD_SIZE, INITIAL_FEN, SQS};

#[wasm_bindgen]
pub fn get_initial_board_fen() -> String {
    INITIAL_FEN.to_string()
}

#[wasm_bindgen]
pub fn get_board_size() -> usize {
    BOARD_SIZE
}

#[wasm_bindgen]
pub fn get_total_squares() -> usize {
    SQS
}