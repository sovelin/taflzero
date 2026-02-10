use crate::board::utils::{get_col, get_row};
use crate::mv::Move;

const DIRECTIONS: usize = 4;
const MAX_DISTANCE: usize = 10;

enum Direction {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

fn get_move_direction(mv: Move) -> Direction {
    let from = mv.from() as usize;
    let to = mv.to() as usize;

    let from_row = get_row(from);
    let to_row = get_row(to);
    let from_col = get_col(from);
    let to_col = get_col(to);

    if from_row == to_row {
        if to_col > from_col {
            Direction::Right
        } else {
            Direction::Left
        }
    } else {
        if to_row > from_row {
            Direction::Down
        } else {
            Direction::Up
        }
    }
}

fn get_distance(mv: Move) -> usize {
    let from = mv.from() as usize;
    let to = mv.to() as usize;

    let from_row = get_row(from);
    let to_row = get_row(to);
    let from_col = get_col(from);
    let to_col = get_col(to);

    ((from_row as isize - to_row as isize).abs() + (from_col as isize - to_col as isize).abs()) as usize
}

fn move_to_policy_index(mv: Move) -> usize {
    let from = mv.from() as usize;
    let direction = get_move_direction(mv) as usize;

    from * DIRECTIONS * MAX_DISTANCE + direction * MAX_DISTANCE + get_distance(mv)
}