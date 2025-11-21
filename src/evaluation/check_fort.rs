use std::collections::{HashSet, VecDeque};
use crate::board::{Board, PRECOMPUTED};
use crate::board::constants::{BOARD_SIZE, HOLE, SQS};
use crate::board::types::{Piece, Square};
use crate::board::utils::is_edge_square;

pub fn check_fort(board: &mut Board) -> bool {
    if board.king_sq == HOLE {
        return false;
    }

    if !is_calculate_needed(board) {
        return false;
    }

    if !king_has_moves(board) || !king_contacts_edge(board.king_sq as Square) {
        return false;
    }

    let fort_info = get_fort(board);

    if fort_info.is_attackers_accessing_king || fort_info.is_corner_accessing_king {
        return false;
    }

    if fort_info.fort_set.is_empty() {
        return false;
    }

    !is_fort_breakable(board, &fort_info)
}

struct FortInfo {
    fort_list: Vec<Square>,
    fort_set: HashSet<Square>,
    is_attackers_accessing_king: bool,
    is_corner_accessing_king: bool,
    inner_space: [bool; SQS],
}

fn is_calculate_needed(board: &Board) -> bool {
    if board.last_move_to == HOLE {
        return true;
    };

    let last_move_piece = board.board[board.last_move_to as usize];

    if last_move_piece == Piece::KING {
        return true;
    }

    if last_move_piece == Piece::ATTACKER {
        return false;
    }

    let is_edge = is_edge_square(board.last_move_to as usize);
    let neighbors = &PRECOMPUTED.all_neighbors[board.last_move_to as usize];

    let mut defenders_nearby = 0;
    for &sq in neighbors.iter() {
        if board.board[sq] == Piece::DEFENDER || is_edge_square(sq) {
            defenders_nearby += 1;
        }
    }

    (is_edge && defenders_nearby > 0) || defenders_nearby >= 2
}

fn king_has_moves(board: &Board) -> bool {
    for &neighbor in PRECOMPUTED.vertical_horizontal_neighbors[board.king_sq as usize].iter() {
        if board.board[neighbor] == Piece::EMPTY {
            return true;
        }
    }
    false
}

fn king_contacts_edge(king_sq: Square) -> bool {
    let row = PRECOMPUTED.row[king_sq];
    let col = PRECOMPUTED.col[king_sq];

    row == 0 || row == BOARD_SIZE - 1 || col == 0 || col == BOARD_SIZE - 1
}

fn get_fort(board: &Board) -> FortInfo {
    let mut fort_list = Vec::new();
    let mut fort_set: HashSet<Square> = HashSet::new();
    let mut is_attackers_accessing_king = false;
    let mut is_corner_accessing_king = false;

    let mut inner_space = bfs_ts(
        &[board.king_sq as Square],
        &PRECOMPUTED.vertical_horizontal_neighbors,
        |sq| {
            let piece = board.board[sq];

            if piece == Piece::ATTACKER {
                is_attackers_accessing_king = true;
            }

            if is_corner_square(sq) {
                is_corner_accessing_king = true;
            }

            if piece == Piece::DEFENDER && fort_set.insert(sq) {
                fort_list.push(sq);
            }

            piece == Piece::EMPTY
        },
    );

    inner_space[board.king_sq as usize] = true;

    FortInfo {
        fort_list,
        fort_set,
        is_attackers_accessing_king,
        is_corner_accessing_king,
        inner_space,
    }
}

fn is_fort_breakable(board: &mut Board, fort_info: &FortInfo) -> bool {
    let Some(&fort_sq) = fort_info
        .fort_list
        .iter()
        .find(|&&sq| board.board[sq] == Piece::DEFENDER) else {
        return false;
    };

    let connected_defenders = bfs_ts(
        &[fort_sq],
        &PRECOMPUTED.all_neighbors,
        |sq| board.board[sq] == Piece::DEFENDER,
    );

    for sq in flags_to_vec(&connected_defenders) {
        let left = PRECOMPUTED.left_neighbor[sq];
        let right = PRECOMPUTED.right_neighbor[sq];

        if is_capture_possible_in_enemy_area(board, left, right, &fort_info.inner_space) {
            if fort_info.fort_set.contains(&sq) {
                return true;
            }

            board.clear_piece(sq);
            let res = is_fort_breakable(board, fort_info);
            board.set_piece(sq, Piece::DEFENDER);

            if res {
                return true;
            }
        }

        let top = PRECOMPUTED.top_neighbor[sq];
        let bottom = PRECOMPUTED.bottom_neighbor[sq];

        if is_capture_possible_in_enemy_area(board, top, bottom, &fort_info.inner_space) {
            if fort_info.fort_set.contains(&sq) {
                return true;
            }

            board.clear_piece(sq);
            let res = is_fort_breakable(board, fort_info);
            board.set_piece(sq, Piece::DEFENDER);

            if res {
                return true;
            }
        }
    }

    false
}

fn is_capture_possible_in_enemy_area(
    board: &Board,
    between_a: Option<Square>,
    between_b: Option<Square>,
    inner_space: &[bool; SQS],
) -> bool {
    let (Some(a), Some(b)) = (between_a, between_b) else {
        return false;
    };

    is_from_outside(a, b, inner_space)
        && could_have_potential_attacker(board, a)
        && could_have_potential_attacker(board, b)
}

fn is_from_outside(a: Square, b: Square, inner_space: &[bool; SQS]) -> bool {
    !inner_space[a] && !inner_space[b]
}

fn could_have_potential_attacker(board: &Board, sq: Square) -> bool {
    board.board[sq] != Piece::DEFENDER && sq != PRECOMPUTED.throne_sq
}

fn is_corner_square(sq: Square) -> bool {
    sq == PRECOMPUTED.top_left_sq
        || sq == PRECOMPUTED.bottom_left_sq
        || sq == PRECOMPUTED.top_right_sq
        || sq == PRECOMPUTED.bottom_right_sq
}

fn bfs_ts<F>(start_squares: &[Square], neighbors: &[Vec<Square>; SQS], mut is_achievable: F) -> [bool; SQS]
where
    F: FnMut(Square) -> bool,
{
    let mut queue: VecDeque<Square> = VecDeque::new();
    let mut visited = [false; SQS];
    let mut flags = [false; SQS];

    for &sq in start_squares {
        queue.push_back(sq);
    }

    while let Some(current) = queue.pop_front() {
        for &neighbor in neighbors[current].iter() {
            if visited[neighbor] {
                continue;
            }

            if !is_achievable(neighbor) {
                continue;
            }

            visited[neighbor] = true;
            flags[neighbor] = true;
            queue.push_back(neighbor);
        }
    }

    flags
}

fn flags_to_vec(flags: &[bool; SQS]) -> Vec<Square> {
    flags
        .iter()
        .enumerate()
        .filter_map(|(idx, &flag)| if flag { Some(idx) } else { None })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::board::types::Piece;
    use crate::board::utils::get_square_from_algebraic;
    use super::check_fort;

    #[test]
    fn empty_king_not_fort() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("f6"), Piece::KING).unwrap();
        assert!(!check_fort(&mut board));
    }

    #[test]
    fn king_surrounded_by_defenders_in_center_not_fort() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("f6"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("e6"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("e7"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("g6"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("g7"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("f5"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("f8"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("e5"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("g5"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("g8"), Piece::DEFENDER).unwrap();
        assert!(!check_fort(&mut board));
    }

    #[test]
    fn king_surrounded_on_edge_impossible_to_break_king_has_moves_fort() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("f1"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("e1"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("e2"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("g1"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("g2"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("f3"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("e3"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("g3"), Piece::DEFENDER).unwrap();
        assert!(check_fort(&mut board));
    }

    #[test]
    fn king_surrounded_have_access_to_edge_not_on_edge_not_fort() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("f2"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("e1"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("e2"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("g1"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("g2"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("f3"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("e3"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("g3"), Piece::DEFENDER).unwrap();
        assert!(!check_fort(&mut board));
    }

    #[test]
    fn king_surrounded_on_edge_attacker_inside_not_fort() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("f1"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("e1"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("e2"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("g1"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("g2"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("f4"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("e3"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("g3"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("e4"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("g4"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("f3"), Piece::ATTACKER).unwrap();
        assert!(!check_fort(&mut board));
    }

    #[test]
    fn king_surrounded_by_attackers_like_fort_not_fort() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("f2"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("e1"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("e2"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("g1"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("g2"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("f3"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("e3"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("g3"), Piece::ATTACKER).unwrap();
        assert!(!check_fort(&mut board));
    }

    #[test]
    fn king_surrounded_on_edge_no_moves_not_fort() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("f1"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("e1"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("g1"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("f2"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("e2"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("g2"), Piece::DEFENDER).unwrap();
        assert!(!check_fort(&mut board));
    }

    #[test]
    fn king_surrounded_on_edge_possible_to_break_no_moves_not_fort() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("f1"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("e1"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("g1"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("f3"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("e2"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("g2"), Piece::DEFENDER).unwrap();
        assert!(!check_fort(&mut board));
    }

    #[test]
    fn king_surrounded_on_edge_impossible_to_break_no_moves_complex_case_fort() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("f1"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("e1"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("g1"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("f3"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("e2"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("g2"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("e3"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("f4"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("g3"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("d2"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("h2"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("i2"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("c2"), Piece::DEFENDER).unwrap();
        assert!(check_fort(&mut board));
    }
}
