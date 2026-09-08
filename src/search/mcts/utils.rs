use crate::board::utils::{get_col, get_row};
#[cfg(test)]
use crate::board::{types::Square, utils::get_square};
use crate::mv::Move;

// TODO(board-size): the policy encoding is inherently size-dependent (MAX_DISTANCE is
// n-1, POLICY_SIZE is n*n*4*(n-1)), and these helpers have no board to ask. Hardcoded
// to 11 until the policy layout itself is parameterized.
const DIRECTIONS: u16 = 4;
const MAX_DISTANCE: u16 = 10;

enum Direction {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

fn get_move_direction(mv: Move) -> Direction {
    let from = mv.from();
    let to = mv.to();

    let from_row = get_row(from, 11);
    let to_row = get_row(to, 11);
    let from_col = get_col(from, 11);
    let to_col = get_col(to, 11);

    if from_row == to_row {
        if to_col > from_col {
            Direction::Right
        } else {
            Direction::Left
        }
    } else if to_row > from_row {
        Direction::Down
    } else {
        Direction::Up
    }
}

fn get_distance(mv: Move) -> usize {
    let from = mv.from();
    let to = mv.to();

    let from_row = get_row(from, 11);
    let to_row = get_row(to, 11);
    let from_col = get_col(from, 11);
    let to_col = get_col(to, 11);

    ((from_row as isize - to_row as isize).abs() + (from_col as isize - to_col as isize).abs())
        as usize
}

pub fn move_to_policy_index(mv: Move) -> u16 {
    let from = mv.from() as u16;
    let direction = get_move_direction(mv) as u16;

    // distance ∈ [1..MAX_DISTANCE]
    let distance = get_distance(mv) as u16;

    debug_assert!((1..=MAX_DISTANCE).contains(&distance));

    from * (DIRECTIONS * MAX_DISTANCE) + direction * MAX_DISTANCE + (distance - 1)
}

#[cfg(test)]
pub fn policy_index_to_move(index: u16) -> Option<Move> {
    let from = index / (DIRECTIONS * MAX_DISTANCE);
    let direction = (index / MAX_DISTANCE) % DIRECTIONS;
    let distance = (index % MAX_DISTANCE) + 1; // <-- revert 1..=MAX_DISTANCE

    let from_row = get_row(from as usize, 11);
    let from_col = get_col(from as usize, 11);

    let (to_row, to_col) = match direction {
        0 => (from_row as i32 - distance as i32, from_col as i32), // Up
        1 => (from_row as i32 + distance as i32, from_col as i32), // Down
        2 => (from_row as i32, from_col as i32 - distance as i32), // Left
        3 => (from_row as i32, from_col as i32 + distance as i32), // Right
        _ => return None,
    };

    if to_row < 0 || to_col < 0 {
        return None;
    }

    let size = 11i32;
    if to_row >= size || to_col >= size {
        return None;
    }

    let to_square = get_square(to_row as usize, to_col as usize, 11);

    Some(Move::new(from as Square, to_square as Square))
}

#[cfg(test)]
mod tests {
    use crate::board::utils::get_square;
    use crate::mv::Move;
    use crate::search::mcts::utils::{move_to_policy_index, policy_index_to_move};

    #[test]
    fn test_move_policy_index_mapping() {
        let moves = [
            Move::new(get_square(0, 0, 11), get_square(0, 1, 11)),
            Move::new(get_square(0, 0, 11), get_square(0, 5, 11)),
            Move::new(get_square(5, 5, 11), get_square(5, 4, 11)),
            Move::new(get_square(5, 5, 11), get_square(4, 5, 11)),
            Move::new(get_square(7, 6, 11), get_square(7, 8, 11)),
            Move::new(get_square(7, 6, 11), get_square(8, 6, 11)),
        ];

        for mv in &moves {
            let index = move_to_policy_index(*mv);
            let decoded_move = policy_index_to_move(index).expect("Decoded move should be valid");
            assert_eq!(*mv, decoded_move);
        }
    }
}
