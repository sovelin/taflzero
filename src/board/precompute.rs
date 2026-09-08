use crate::board::constants::BOARD_SIZE;
use crate::board::masks::{Mask, create_line_moves};
use crate::board::types::{Col, Row, Square};
use crate::board::utils::{
    get_all_neighbors, get_bottom_left_neighbor, get_bottom_left_sq, get_bottom_neighbor,
    get_bottom_right_neighbor, get_bottom_right_sq, get_col, get_corners_sq, get_edges_sq,
    get_left_neighbor, get_right_neighbor, get_row, get_throne_sq, get_top_left_neighbor,
    get_top_left_sq, get_top_neighbor, get_top_right_neighbor, get_top_right_sq,
    get_vertical_horizontal_neighbors,
};
use std::sync::LazyLock;

type CellsArray<T> = Vec<Option<T>>;

pub struct SandwichCapture {
    pub between_sq: Square,
    pub captor_sq: Square,
}

pub struct Precomputed {
    pub board_size: usize,
    pub sqs: usize,
    pub row: Vec<Row>,
    pub col: Vec<Col>,
    pub throne_sq: usize,
    pub corners_sq: Vec<Square>,
    pub edges_sq: Vec<Square>,
    pub top_left_sq: usize,
    pub top_right_sq: usize,
    pub bottom_right_sq: usize,
    pub bottom_left_sq: usize,

    pub right_neighbor: CellsArray<Square>,
    pub left_neighbor: CellsArray<Square>,
    pub top_neighbor: CellsArray<Square>,
    pub bottom_neighbor: CellsArray<Square>,
    pub top_right_neighbor: CellsArray<Square>,
    pub top_left_neighbor: CellsArray<Square>,
    pub bottom_right_neighbor: CellsArray<Square>,
    pub bottom_left_neighbor: CellsArray<Square>,
    pub vertical_horizontal_neighbors: Vec<Vec<Square>>,
    pub all_neighbors: Vec<Vec<Square>>,
    pub manhattan_distance: Vec<Vec<usize>>,
    pub sandwich_captures: Vec<Vec<SandwichCapture>>,
    pub line_moves: Vec<Vec<Mask>>,
}

pub fn get_right_sandwich_capture(sq: Square, board_size: usize) -> Option<SandwichCapture> {
    let col = get_col(sq, board_size);

    if col + 2 >= board_size {
        return None;
    }

    Some(SandwichCapture {
        between_sq: sq + 1,
        captor_sq: sq + 2,
    })
}

pub fn get_left_sandwich_capture(sq: Square, board_size: usize) -> Option<SandwichCapture> {
    let col = get_col(sq, board_size);

    if col < 2 {
        return None;
    }

    Some(SandwichCapture {
        between_sq: sq - 1,
        captor_sq: sq - 2,
    })
}

pub fn get_up_sandwich_capture(sq: Square, board_size: usize) -> Option<SandwichCapture> {
    let row = get_row(sq, board_size);

    if row + 2 >= board_size {
        return None;
    }

    Some(SandwichCapture {
        between_sq: sq + board_size,
        captor_sq: sq + 2 * board_size,
    })
}

pub fn get_down_sandwich_capture(sq: Square, board_size: usize) -> Option<SandwichCapture> {
    let row = get_row(sq, board_size);

    if row < 2 {
        return None;
    }

    Some(SandwichCapture {
        between_sq: sq - board_size,
        captor_sq: sq - 2 * board_size,
    })
}

pub fn precompute_sandwich_captures(vec: &mut Vec<SandwichCapture>, sq: Square, board_size: usize) {
    if let Some(capture) = get_right_sandwich_capture(sq, board_size) {
        vec.push(capture);
    }
    if let Some(capture) = get_left_sandwich_capture(sq, board_size) {
        vec.push(capture);
    }
    if let Some(capture) = get_up_sandwich_capture(sq, board_size) {
        vec.push(capture);
    }
    if let Some(capture) = get_down_sandwich_capture(sq, board_size) {
        vec.push(capture);
    }
}

impl Default for Precomputed {
    fn default() -> Self {
        Self::new(BOARD_SIZE)
    }
}

impl Precomputed {
    pub fn new(board_size: usize) -> Self {
        let sqs = board_size * board_size;

        let mut row: Vec<Row> = Vec::with_capacity(sqs);
        let mut col: Vec<Col> = Vec::with_capacity(sqs);
        let mut right_neighbor: CellsArray<Square> = Vec::with_capacity(sqs);
        let mut left_neighbor: CellsArray<Square> = Vec::with_capacity(sqs);
        let mut top_neighbor: CellsArray<Square> = Vec::with_capacity(sqs);
        let mut bottom_neighbor: CellsArray<Square> = Vec::with_capacity(sqs);
        let mut top_right_neighbor: CellsArray<Square> = Vec::with_capacity(sqs);
        let mut top_left_neighbor: CellsArray<Square> = Vec::with_capacity(sqs);
        let mut bottom_right_neighbor: CellsArray<Square> = Vec::with_capacity(sqs);
        let mut bottom_left_neighbor: CellsArray<Square> = Vec::with_capacity(sqs);
        let mut vertical_horizontal_neighbors: Vec<Vec<Square>> = Vec::with_capacity(sqs);
        let mut all_neighbors: Vec<Vec<Square>> = Vec::with_capacity(sqs);
        let mut manhattan_distance: Vec<Vec<usize>> = vec![vec![0; sqs]; sqs];
        let mut sandwich_captures: Vec<Vec<SandwichCapture>> =
            (0..sqs).map(|_| Vec::new()).collect();

        for i in 0..sqs {
            row.push(get_row(i, board_size));
            col.push(get_col(i, board_size));
            left_neighbor.push(get_left_neighbor(i, board_size));
            right_neighbor.push(get_right_neighbor(i, board_size));
            top_neighbor.push(get_top_neighbor(i, board_size));
            bottom_neighbor.push(get_bottom_neighbor(i, board_size));
            top_right_neighbor.push(get_top_right_neighbor(i, board_size));
            top_left_neighbor.push(get_top_left_neighbor(i, board_size));
            bottom_right_neighbor.push(get_bottom_right_neighbor(i, board_size));
            bottom_left_neighbor.push(get_bottom_left_neighbor(i, board_size));
            vertical_horizontal_neighbors.push(get_vertical_horizontal_neighbors(i, board_size));
            all_neighbors.push(get_all_neighbors(i, board_size));
            precompute_sandwich_captures(&mut sandwich_captures[i], i, board_size);
        }

        for i in 0..sqs {
            for j in 0..sqs {
                let row_diff = row[i].abs_diff(row[j]);
                let col_diff = col[i].abs_diff(col[j]);
                manhattan_distance[i][j] = row_diff + col_diff;
            }
        }

        Self {
            board_size,
            sqs,
            row,
            col,
            left_neighbor,
            right_neighbor,
            top_neighbor,
            bottom_neighbor,
            top_right_neighbor,
            top_left_neighbor,
            bottom_right_neighbor,
            bottom_left_neighbor,
            vertical_horizontal_neighbors,
            all_neighbors,
            manhattan_distance,
            throne_sq: get_throne_sq(board_size),
            corners_sq: get_corners_sq(board_size),
            edges_sq: get_edges_sq(board_size),
            top_left_sq: get_top_left_sq(board_size),
            top_right_sq: get_top_right_sq(board_size),
            bottom_right_sq: get_bottom_right_sq(board_size),
            bottom_left_sq: get_bottom_left_sq(board_size),
            sandwich_captures,
            line_moves: create_line_moves(board_size),
        }
    }
}

// TODO(board-size): single global instance fixed at BOARD_SIZE. Supporting several
// sizes at once means a per-size registry plus a reference on Board, not a static.
pub static PRECOMPUTED: LazyLock<Precomputed> = LazyLock::new(|| Precomputed::new(BOARD_SIZE));

#[cfg(test)]
mod tests {
    use super::*;

    // Precomputed::new must derive every table from its argument, not from BOARD_SIZE.
    #[test]
    fn geometry_follows_board_size() {
        let p = Precomputed::new(9);

        assert_eq!(p.board_size, 9);
        assert_eq!(p.sqs, 81);
        assert_eq!(p.row.len(), 81);
        assert_eq!(p.col.len(), 81);

        assert_eq!(p.throne_sq, 40); // (4, 4) on a 9x9 board
        assert_eq!(p.corners_sq, vec![0, 8, 72, 80]);
        assert_eq!(p.edges_sq.len(), 32); // 9*9 - 7*7
        assert_eq!(p.bottom_left_sq, 0);
        assert_eq!(p.top_right_sq, 80);

        assert_eq!(p.right_neighbor[7], Some(8));
        assert_eq!(p.right_neighbor[8], None); // right edge
        assert_eq!(p.top_neighbor[80], None); // top edge
        assert_eq!(p.sandwich_captures[0].len(), 2); // corner: right + up only

        assert_eq!(p.line_moves.len(), 9);
        assert_eq!(p.line_moves[0].len(), 1 << 9);
    }
}
