use crate::board::constants::{BOARD_SIZE, SQS};
use crate::board::types::{Col, Row, Square};
use crate::board::utils::{
    get_all_neighbors, get_bottom_left_neighbor, get_bottom_left_sq, get_bottom_neighbor,
    get_bottom_right_neighbor, get_bottom_right_sq, get_col, get_corners_sq, get_left_neighbor,
    get_right_neighbor, get_row, get_throne_sq, get_top_left_neighbor, get_top_left_sq,
    get_top_neighbor, get_top_right_neighbor, get_top_right_sq, get_vertical_horizontal_neighbors,
};
use std::sync::LazyLock;

type CellsArray<T> = [Option<T>; SQS];

pub struct SandwichCapture {
    pub between_sq: Square,
    pub captor_sq: Square,
}

pub struct Precomputed {
    pub row: [Row; SQS],
    pub col: [Col; SQS],
    pub throne_sq: usize,
    pub corners_sq: Vec<Square>,
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
    pub vertical_horizontal_neighbors: [Vec<Square>; SQS],
    pub all_neighbors: [Vec<Square>; SQS],
    pub manhattan_distance: [[usize; SQS]; SQS],
    pub sandwich_captures: [Vec<SandwichCapture>; SQS],
}

pub fn get_right_sandwich_capture(sq: Square) -> Option<SandwichCapture> {
    let col = get_col(sq);

    if col + 2 >= BOARD_SIZE {
        return None;
    }

    Some(SandwichCapture {
        between_sq: sq + 1,
        captor_sq: sq + 2,
    })
}

pub fn get_left_sandwich_capture(sq: Square) -> Option<SandwichCapture> {
    let col = get_col(sq);

    if col < 2 {
        return None;
    }

    Some(SandwichCapture {
        between_sq: sq - 1,
        captor_sq: sq - 2,
    })
}

pub fn get_up_sandwich_capture(sq: Square) -> Option<SandwichCapture> {
    let row = get_row(sq);

    if row + 2 >= BOARD_SIZE {
        return None;
    }

    Some(SandwichCapture {
        between_sq: sq + BOARD_SIZE,
        captor_sq: sq + 2 * BOARD_SIZE,
    })
}

pub fn get_down_sandwich_capture(sq: Square) -> Option<SandwichCapture> {
    let row = get_row(sq);

    if row < 2 {
        return None;
    }

    Some(SandwichCapture {
        between_sq: sq - BOARD_SIZE,
        captor_sq: sq - 2 * BOARD_SIZE,
    })
}

pub fn precompute_sandwich_captures(vec: &mut Vec<SandwichCapture>, sq: Square) {
    if let Some(capture) = get_right_sandwich_capture(sq) {
        vec.push(capture);
    }
    if let Some(capture) = get_left_sandwich_capture(sq) {
        vec.push(capture);
    }
    if let Some(capture) = get_up_sandwich_capture(sq) {
        vec.push(capture);
    }
    if let Some(capture) = get_down_sandwich_capture(sq) {
        vec.push(capture);
    }
}

impl Precomputed {
    pub fn new() -> Self {
        let mut col = [0; SQS];
        let mut row = [0; SQS];
        let mut right_neighbor: CellsArray<Square> = [None; SQS];
        let mut left_neighbor: CellsArray<Square> = [None; SQS];
        let mut top_neighbor: CellsArray<Square> = [None; SQS];
        let mut bottom_neighbor: CellsArray<Square> = [None; SQS];
        let mut top_right_neighbor: CellsArray<Square> = [None; SQS];
        let mut top_left_neighbor: CellsArray<Square> = [None; SQS];
        let mut bottom_right_neighbor: CellsArray<Square> = [None; SQS];
        let mut bottom_left_neighbor: CellsArray<Square> = [None; SQS];
        let mut vertical_horizontal_neighbors: [Vec<Square>; SQS] =
            std::array::from_fn(|_| Vec::new());
        let mut all_neighbors: [Vec<Square>; SQS] = std::array::from_fn(|_| Vec::new());
        let mut manhattan_distance = [[0; SQS]; SQS];
        let mut sandwich_captures: [Vec<SandwichCapture>; SQS] =
            std::array::from_fn(|_| Vec::new());

        for i in 0..SQS {
            row[i] = get_row(i);
            col[i] = get_col(i);
            left_neighbor[i] = get_left_neighbor(i);
            right_neighbor[i] = get_right_neighbor(i);
            top_neighbor[i] = get_top_neighbor(i);
            bottom_neighbor[i] = get_bottom_neighbor(i);
            top_right_neighbor[i] = get_top_right_neighbor(i);
            top_left_neighbor[i] = get_top_left_neighbor(i);
            bottom_right_neighbor[i] = get_bottom_right_neighbor(i);
            bottom_left_neighbor[i] = get_bottom_left_neighbor(i);
            vertical_horizontal_neighbors[i] = get_vertical_horizontal_neighbors(i);
            all_neighbors[i] = get_all_neighbors(i);
            precompute_sandwich_captures(&mut sandwich_captures[i], i);
        }

        for i in 0..SQS {
            for j in 0..SQS {
                let row_diff = if row[i] > row[j] {
                    row[i] - row[j]
                } else {
                    row[j] - row[i]
                };
                let col_diff = if col[i] > col[j] {
                    col[i] - col[j]
                } else {
                    col[j] - col[i]
                };
                manhattan_distance[i][j] = row_diff + col_diff;
            }
        }

        Self {
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
            throne_sq: get_throne_sq(),
            corners_sq: get_corners_sq(),
            top_left_sq: get_top_left_sq(),
            top_right_sq: get_top_right_sq(),
            bottom_right_sq: get_bottom_right_sq(),
            bottom_left_sq: get_bottom_left_sq(),
            sandwich_captures,
        }
    }
}

pub static PRECOMPUTED: LazyLock<Precomputed> = LazyLock::new(Precomputed::new);
