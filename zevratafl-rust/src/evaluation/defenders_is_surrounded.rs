use crate::board::{Board, PRECOMPUTED};
use crate::board::constants::HOLE;
use crate::board::types::{Piece, Square};
use crate::board::utils::is_edge_square;
use crate::utils::bfs::bfs;

fn is_calculate_needed(board: &Board) -> bool {
    if board.last_move_to == HOLE {
        return true;
    }

    if board.board[board.last_move_to as usize] != Piece::ATTACKER {
        return false;
    }

    let all_neighbors = &PRECOMPUTED.vertical_horizontal_neighbors[board.last_move_to as usize];

    let mut attackers_nearby_count = 0;

    for &neighbor in all_neighbors.iter() {
        if board.board[neighbor] == Piece::ATTACKER {
            attackers_nearby_count += 1;
        }

        if attackers_nearby_count >= 2 {
            return true;
        }
    }

    false
}

pub fn defenders_is_surrounded(board: &Board) -> bool {
    if !is_calculate_needed(board) {
        return false;
    }

    let visited = bfs(
        |sq| {
            board.board[sq] != Piece::ATTACKER
        },
        &PRECOMPUTED.vertical_horizontal_neighbors,
        &PRECOMPUTED.corners_sq,
    );

    for i in 0..visited.len() {
        if !visited[i] {
            continue;
        }

        if board.board[i] == Piece::DEFENDER || board.board[i] == Piece::KING {
            return false;
        }
    }

    let inner_squares = bfs(
        |sq| {
            board.board[sq] != Piece::ATTACKER
        },
        &PRECOMPUTED.vertical_horizontal_neighbors,
        &vec![board.king_sq as Square],
    );

    for i in 0..inner_squares.len() {
        if inner_squares[i] && is_edge_square(i as Square) {
            return false
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::board::types::Piece;
    use crate::board::utils::get_square_from_algebraic;
    use super::defenders_is_surrounded;

    #[test]
    fn by_default_not_surrounded() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("d4"), Piece::KING).unwrap();

        let is_surrounded = defenders_is_surrounded(&board);
        assert!(!is_surrounded);
    }

    #[test]
    fn surrounded_corners_but_with_open_edges_not_surrounded() {
        let mut board = Board::new();

        board.set_piece(get_square_from_algebraic("a2"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b1"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("j1"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("k2"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("a10"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b11"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("j11"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("k10"), Piece::ATTACKER).unwrap();

        board.set_piece(get_square_from_algebraic("d4"), Piece::KING).unwrap();

        let is_surrounded = defenders_is_surrounded(&board);
        assert!(!is_surrounded);
    }

    #[test]
    fn surrounded_corners_and_edges_no_escape_surrounded() {
        let mut board = Board::new();

        board.set_piece(get_square_from_algebraic("a10"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b11"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("c11"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("d10"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("c9"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b9"), Piece::ATTACKER).unwrap();

        board.set_piece(get_square_from_algebraic("b10"), Piece::KING).unwrap();

        let is_surrounded = defenders_is_surrounded(&board);
        assert!(is_surrounded);
    }

    #[test]
    fn surrounded_corners_and_edges_one_defender_outside_not_surrounded() {
        let mut board = Board::new();

        board.set_piece(get_square_from_algebraic("a10"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b11"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("c11"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("d10"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("c9"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b9"), Piece::ATTACKER).unwrap();

        board.set_piece(get_square_from_algebraic("b10"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("j1"), Piece::DEFENDER).unwrap();

        let is_surrounded = defenders_is_surrounded(&board);
        assert!(!is_surrounded);
    }

    #[test]
    fn surrounded_corners_and_edges_but_defender_in_chain_not_surrounded() {
        let mut board = Board::new();

        board.set_piece(get_square_from_algebraic("a10"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b11"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("c11"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("d10"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("c9"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b9"), Piece::DEFENDER).unwrap();

        board.set_piece(get_square_from_algebraic("b10"), Piece::KING).unwrap();

        let is_surrounded = defenders_is_surrounded(&board);
        assert!(!is_surrounded);
    }
}
