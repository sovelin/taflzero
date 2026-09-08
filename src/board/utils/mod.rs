mod tests;

use crate::board::types::{Col, Piece, Row, Side, Square};
use wasm_bindgen::prelude::wasm_bindgen;

#[wasm_bindgen]
pub fn get_square(row: Row, col: Col, board_size: usize) -> Square {
    row * board_size + col
}

// TODO(board-size): algebraic notation crosses the UCI/JS boundary where no board is
// available, so it stays on the BOARD_SIZE constant — same as `get_sq_algebraic`.
#[wasm_bindgen]
pub fn get_square_from_algebraic(coord: &str) -> Square {
    let file = coord.as_bytes()[0] - b'a';
    let rank = coord[1..].parse::<u8>().unwrap() - 1;

    get_square(rank as Row, file as Col, 11)
}

#[wasm_bindgen]
pub fn get_row(sq: Square, board_size: usize) -> Row {
    sq / board_size
}

#[wasm_bindgen]
pub fn get_col(sq: Square, board_size: usize) -> Col {
    sq % board_size
}

pub fn get_side_by_piece(piece: Piece) -> Result<Side, String> {
    match piece {
        Piece::ATTACKER => Ok(Side::ATTACKERS),
        Piece::DEFENDER => Ok(Side::DEFENDERS),
        Piece::KING => Ok(Side::DEFENDERS),
        Piece::EMPTY => Err("Empty square has no side".to_string()),
    }
}

pub fn get_throne_sq(board_size: usize) -> Square {
    let center = (board_size - 1) / 2;
    get_square(center, center, board_size)
}

pub fn get_corners_sq(board_size: usize) -> Vec<Square> {
    vec![
        get_square(0, 0, board_size),
        get_square(0, board_size - 1, board_size),
        get_square(board_size - 1, 0, board_size),
        get_square(board_size - 1, board_size - 1, board_size),
    ]
}

pub fn get_top_left_sq(board_size: usize) -> Square {
    get_square(board_size - 1, 0, board_size)
}

pub fn get_top_right_sq(board_size: usize) -> Square {
    get_square(board_size - 1, board_size - 1, board_size)
}

pub fn get_bottom_left_sq(board_size: usize) -> Square {
    get_square(0, 0, board_size)
}

pub fn get_bottom_right_sq(board_size: usize) -> Square {
    get_square(0, board_size - 1, board_size)
}

pub fn get_right_neighbor(sq: Square, board_size: usize) -> Option<Square> {
    let col = get_col(sq, board_size);
    if col + 1 >= board_size {
        return None;
    }
    Some(sq + 1)
}

pub fn get_left_neighbor(sq: Square, board_size: usize) -> Option<Square> {
    let col = get_col(sq, board_size);
    if col == 0 {
        return None;
    }
    Some(sq - 1)
}

pub fn get_top_neighbor(sq: Square, board_size: usize) -> Option<Square> {
    let row = get_row(sq, board_size);
    if row + 1 >= board_size {
        return None;
    }
    Some(sq + board_size)
}

pub fn get_bottom_neighbor(sq: Square, board_size: usize) -> Option<Square> {
    let row = get_row(sq, board_size);
    if row == 0 {
        return None;
    }
    Some(sq - board_size)
}

pub fn get_top_left_neighbor(sq: Square, board_size: usize) -> Option<Square> {
    let row = get_row(sq, board_size);
    let col = get_col(sq, board_size);
    if row + 1 >= board_size || col == 0 {
        return None;
    }
    Some(sq + board_size - 1)
}

pub fn get_top_right_neighbor(sq: Square, board_size: usize) -> Option<Square> {
    let row = get_row(sq, board_size);
    let col = get_col(sq, board_size);
    if row + 1 >= board_size || col + 1 >= board_size {
        return None;
    }
    Some(sq + board_size + 1)
}

pub fn get_bottom_left_neighbor(sq: Square, board_size: usize) -> Option<Square> {
    let row = get_row(sq, board_size);
    let col = get_col(sq, board_size);
    if row == 0 || col == 0 {
        return None;
    }
    Some(sq - board_size - 1)
}

pub fn get_bottom_right_neighbor(sq: Square, board_size: usize) -> Option<Square> {
    let row = get_row(sq, board_size);
    let col = get_col(sq, board_size);
    if row == 0 || col + 1 >= board_size {
        return None;
    }
    Some(sq - board_size + 1)
}

pub fn get_vertical_horizontal_neighbors(sq: Square, board_size: usize) -> Vec<Square> {
    let mut neighbors = Vec::new();

    if let Some(n) = get_top_neighbor(sq, board_size) {
        neighbors.push(n);
    }
    if let Some(n) = get_bottom_neighbor(sq, board_size) {
        neighbors.push(n);
    }
    if let Some(n) = get_left_neighbor(sq, board_size) {
        neighbors.push(n);
    }
    if let Some(n) = get_right_neighbor(sq, board_size) {
        neighbors.push(n);
    }

    neighbors
}

pub fn get_all_neighbors(sq: Square, board_size: usize) -> Vec<Square> {
    let mut neighbors = get_vertical_horizontal_neighbors(sq, board_size);

    if let Some(n) = get_top_left_neighbor(sq, board_size) {
        neighbors.push(n);
    }
    if let Some(n) = get_top_right_neighbor(sq, board_size) {
        neighbors.push(n);
    }
    if let Some(n) = get_bottom_left_neighbor(sq, board_size) {
        neighbors.push(n);
    }
    if let Some(n) = get_bottom_right_neighbor(sq, board_size) {
        neighbors.push(n);
    }

    neighbors
}

pub fn get_edges_sq(board_size: usize) -> Vec<Square> {
    (0..board_size * board_size)
        .filter(|&sq| is_edge_square(sq, board_size))
        .collect()
}

pub fn is_edge_square(sq: Square, board_size: usize) -> bool {
    let row = get_row(sq, board_size);
    let col = get_col(sq, board_size);

    row == 0 || row == board_size - 1 || col == 0 || col == board_size - 1
}

// TODO(board-size): stays on the BOARD_SIZE constant because `Display for Move` has
// no access to a board. Threading geometry into move formatting is a separate step.
#[wasm_bindgen]
pub fn get_sq_algebraic(sq: Square) -> String {
    let mut first = b'a';

    let col = get_col(sq, 11);
    let row = get_row(sq, 11);

    first += col as u8;

    let col_str = (first as char).to_string();
    let row_str = (row + 1).to_string();

    format!("{}{}", col_str, row_str)
}
