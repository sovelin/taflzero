use crate::board::constants::BOARD_SIZE;
use crate::board::types::{Col, Piece, Row, Side, Square};

#[inline(always)]
pub fn get_square(row: Row, col: Col) -> Square {
    row * BOARD_SIZE + col
}

pub fn get_square_from_algebraic(coord: &str) -> Square {
    let file = coord.as_bytes()[0] - b'a';
    let rank = coord[1..].parse::<u8>().unwrap() - 1;

    get_square(rank as Row, file as Col)
}

#[inline(always)]
pub fn get_row(sq: Square) -> Row {
    sq / BOARD_SIZE
}

#[inline(always)]
pub fn get_col(sq: Square) -> Col {
    sq % BOARD_SIZE
}


pub fn get_side_by_piece(piece: Piece) -> Result<Side, String> {
    match piece {
        Piece::ATTACKER => Ok(Side::ATTACKERS),
        Piece::DEFENDER => Ok(Side::DEFENDERS),
        Piece::KING => Ok(Side::DEFENDERS),
        Piece::EMPTY => Err("Empty square has no side".to_string()),
    }
}

pub fn get_throne_sq() -> Square {
   get_square(5, 5)
}

pub fn get_corners_sq() -> Vec<Square> {
    vec![
        get_square(0, 0),
        get_square(0, BOARD_SIZE - 1),
        get_square(BOARD_SIZE - 1, 0),
        get_square(BOARD_SIZE - 1, BOARD_SIZE - 1),
    ]
}

pub fn get_top_left_sq() -> Square {
    get_square(BOARD_SIZE - 1, 0)
}

pub fn get_bottom_right_sq() -> Square {
    get_square(0, BOARD_SIZE - 1)
}

pub fn get_bottom_left_sq() -> Square {
    get_square(0, 0)
}

pub fn get_top_right_sq() -> Square {
    get_square(BOARD_SIZE - 1, BOARD_SIZE - 1)
}

pub fn get_right_neighbor(sq: Square) -> Option<Square> {
    let col = get_col(sq);
    if col + 1 >= BOARD_SIZE {
        return None;
    }
    Some(sq + 1)
}

pub fn get_left_neighbor(sq: Square) -> Option<Square> {
    let col = get_col(sq);
    if col == 0 {
        return None;
    }
    Some(sq - 1)
}

pub fn get_top_neighbor(sq: Square) -> Option<Square> {
    let row = get_row(sq);
    if row + 1 >= BOARD_SIZE {
        return None;
    }
    Some(sq + BOARD_SIZE)
}

pub fn get_bottom_neighbor(sq: Square) -> Option<Square> {
    let row = get_row(sq);
    if row == 0 {
        return None;
    }
    Some(sq - BOARD_SIZE)
}

pub fn get_top_left_neighbor(sq: Square) -> Option<Square> {
    let row = get_row(sq);
    let col = get_col(sq);
    if row + 1 >= BOARD_SIZE || col == 0 {
        return None;
    }
    Some(sq + BOARD_SIZE - 1)
}

pub fn get_top_right_neighbor(sq: Square) -> Option<Square> {
    let row = get_row(sq);
    let col = get_col(sq);
    if row + 1 >= BOARD_SIZE || col + 1 >= BOARD_SIZE {
        return None;
    }
    Some(sq + BOARD_SIZE + 1)
}

pub fn get_bottom_left_neighbor(sq: Square) -> Option<Square> {
    let row = get_row(sq);
    let col = get_col(sq);
    if row == 0 || col == 0 {
        return None;
    }
    Some(sq - BOARD_SIZE - 1)
}

pub fn get_bottom_right_neighbor(sq: Square) -> Option<Square> {
    let row = get_row(sq);
    let col = get_col(sq);
    if row == 0 || col + 1 >= BOARD_SIZE {
        return None;
    }
    Some(sq - BOARD_SIZE + 1)
}

pub fn get_vertical_horizontal_neighbors(sq: Square) -> Vec<Square> {
    let mut neighbors = Vec::new();

    if let Some(n) = get_top_neighbor(sq) {
        neighbors.push(n);
    }
    if let Some(n) = get_bottom_neighbor(sq) {
        neighbors.push(n);
    }
    if let Some(n) = get_left_neighbor(sq) {
        neighbors.push(n);
    }
    if let Some(n) = get_right_neighbor(sq) {
        neighbors.push(n);
    }

    neighbors
}

pub fn get_all_neighbors(sq: Square) -> Vec<Square> {
    let mut neighbors = get_vertical_horizontal_neighbors(sq);

    if let Some(n) = get_top_left_neighbor(sq) {
        neighbors.push(n);
    }
    if let Some(n) = get_top_right_neighbor(sq) {
        neighbors.push(n);
    }
    if let Some(n) = get_bottom_left_neighbor(sq) {
        neighbors.push(n);
    }
    if let Some(n) = get_bottom_right_neighbor(sq) {
        neighbors.push(n);
    }

    neighbors
}

pub fn is_edge_square(sq: Square) -> bool {
    let row = get_row(sq);
    let col = get_col(sq);

    row == 0 || row == BOARD_SIZE - 1 || col == 0 || col == BOARD_SIZE - 1
}

pub fn get_sq_algebraic(sq: Square) -> String {
    let mut first = b'a';


    let col = get_col(sq);
    let row = get_row(sq);

    first += col as u8;

    let col_str = (first as char).to_string();
    let row_str = (row + 1).to_string();

    format!("{}{}", col_str, row_str)
}