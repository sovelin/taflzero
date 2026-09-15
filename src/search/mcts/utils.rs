use crate::board::utils::{get_col, get_row};
#[cfg(test)]
use crate::board::{types::Square, utils::get_square};
use crate::mv::Move;

use crate::search::POLICY_DIRECTIONS;

const DIRECTIONS: u16 = POLICY_DIRECTIONS as u16;

/// Longest slide on a board of this size, and therefore the length of the distance
/// axis in the policy encoding.
#[inline]
fn max_distance(board_size: usize) -> u16 {
    (board_size - 1) as u16
}

enum Direction {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

fn get_move_direction(mv: Move, board_size: usize) -> Direction {
    let from = mv.from();
    let to = mv.to();

    let from_row = get_row(from, board_size);
    let to_row = get_row(to, board_size);
    let from_col = get_col(from, board_size);
    let to_col = get_col(to, board_size);

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

fn get_distance(mv: Move, board_size: usize) -> usize {
    let from = mv.from();
    let to = mv.to();

    let from_row = get_row(from, board_size);
    let to_row = get_row(to, board_size);
    let from_col = get_col(from, board_size);
    let to_col = get_col(to, board_size);

    ((from_row as isize - to_row as isize).abs() + (from_col as isize - to_col as isize).abs())
        as usize
}

pub fn move_to_policy_index(mv: Move, board_size: usize) -> u16 {
    let from = mv.from() as u16;
    let direction = get_move_direction(mv, board_size) as u16;
    let max_distance = max_distance(board_size);

    // distance ∈ [1..max_distance]
    let distance = get_distance(mv, board_size) as u16;

    debug_assert!((1..=max_distance).contains(&distance));

    from * (DIRECTIONS * max_distance) + direction * max_distance + (distance - 1)
}

#[cfg(test)]
pub fn policy_index_to_move(index: u16, board_size: usize) -> Option<Move> {
    let max_distance = max_distance(board_size);
    let from = index / (DIRECTIONS * max_distance);
    let direction = (index / max_distance) % DIRECTIONS;
    let distance = (index % max_distance) + 1; // <-- revert 1..=max_distance

    let from_row = get_row(from as usize, board_size);
    let from_col = get_col(from as usize, board_size);

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

    let size = board_size as i32;
    if to_row >= size || to_col >= size {
        return None;
    }

    let to_square = get_square(to_row as usize, to_col as usize, board_size);

    Some(Move::new(from as Square, to_square as Square))
}

#[cfg(test)]
mod tests {
    use crate::board::utils::get_square;
    use crate::get_policy_size;
    use crate::mv::Move;
    use crate::search::mcts::utils::{move_to_policy_index, policy_index_to_move};

    // Every legal slide must survive encode -> decode, and land inside the policy head
    // sized for that board.
    #[test]
    fn policy_index_round_trips_for_every_slide() {
        for board_size in [9usize, 11] {
            let policy_size = get_policy_size(board_size) as u16;

            for row in 0..board_size {
                for col in 0..board_size {
                    let from = get_square(row, col, board_size);

                    for (to_row, to_col) in (0..board_size)
                        .map(|c| (row, c))
                        .chain((0..board_size).map(|r| (r, col)))
                    {
                        if (to_row, to_col) == (row, col) {
                            continue;
                        }

                        let mv = Move::new(from, get_square(to_row, to_col, board_size));
                        let index = move_to_policy_index(mv, board_size);

                        assert!(
                            index < policy_size,
                            "index {index} outside a {board_size}x{board_size} policy head                              of {policy_size}"
                        );
                        assert_eq!(
                            policy_index_to_move(index, board_size),
                            Some(mv),
                            "round trip on a {board_size}x{board_size} board"
                        );
                    }
                }
            }
        }
    }

    // The 11x11 layout must stay bit-identical: existing nets and selfplay data depend
    // on it.
    #[test]
    fn policy_layout_for_11x11_is_unchanged() {
        assert_eq!(get_policy_size(11), 4840);

        // from * 40 + direction * 10 + (distance - 1)
        let mv = Move::new(get_square(0, 0, 11), get_square(0, 1, 11));
        assert_eq!(move_to_policy_index(mv, 11), 3 * 10); // Right, distance 1

        let mv = Move::new(get_square(5, 5, 11), get_square(4, 5, 11));
        assert_eq!(move_to_policy_index(mv, 11), 60 * 40); // Up, distance 1
    }
}
