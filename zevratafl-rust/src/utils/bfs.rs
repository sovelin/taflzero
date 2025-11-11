use std::collections::{VecDeque};
use crate::board::constants::SQS;
use crate::board::types::Square;

pub fn bfs(
    is_archivable: impl Fn(Square) -> bool,
    neighbors: &[Vec<Square>; SQS],
    start_sq: &Vec<Square>,
) -> [bool; SQS] {
    let mut queue: VecDeque<Square> = VecDeque::new();

    for &sq in start_sq.iter() {
        queue.push_back(sq);
    }

    let mut visited = vec![false; SQS];
    let mut result: [bool; SQS] = [false; SQS];

    while let Some(current_sq) = queue.pop_front() {
        if visited[current_sq] {
            continue;
        }

        visited[current_sq] = true;

        if is_archivable(current_sq) {
            result[current_sq] = true;

            for &neighbor in neighbors[current_sq].iter() {
                if !visited[neighbor] {
                    queue.push_back(neighbor);
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use crate::board::{Board, PRECOMPUTED};
    use crate::board::constants::SQS;
    use crate::board::types::{Piece, Square};
    use crate::board::utils::get_square_from_algebraic;
    use super::bfs;

    fn expect_cell_in_result(res: &[bool; SQS], sq: Square) {
        assert!(
            res[sq],
            "Expected square {} to be in the result set",
            sq
        );
    }

    fn expect_res_size(res: &[bool; SQS], expected_size: usize) {
        let actual_size = res.iter().filter(|&&v| v).count();
        assert_eq!(
            actual_size, expected_size,
            "Expected result size {}, got {}",
            expected_size, actual_size
        );
    }

    #[test]
    fn go_to_cornets_from_restricted_areas() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("d4"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("d3"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("d2"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("d1"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("c4"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b4"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("a4"), Piece::ATTACKER).unwrap();

        let res = bfs(
            |sq| board.board[sq] == Piece::EMPTY,
            &PRECOMPUTED.vertical_horizontal_neighbors,
            &vec![get_square_from_algebraic("a1")],
        );

        expect_res_size(&res, 9);
        expect_cell_in_result(&res, get_square_from_algebraic("a1"));
        expect_cell_in_result(&res, get_square_from_algebraic("a2"));
        expect_cell_in_result(&res, get_square_from_algebraic("a3"));
        expect_cell_in_result(&res, get_square_from_algebraic("b1"));
        expect_cell_in_result(&res, get_square_from_algebraic("b2"));
        expect_cell_in_result(&res, get_square_from_algebraic("b3"));
        expect_cell_in_result(&res, get_square_from_algebraic("c1"));
        expect_cell_in_result(&res, get_square_from_algebraic("c2"));
        expect_cell_in_result(&res, get_square_from_algebraic("c3"));
    }

    #[test]
    fn go_from_different_corners_and_one_restricted() {
        let mut board = Board::new();

        board.set_piece(get_square_from_algebraic("d4"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("d3"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("d2"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("d1"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("c4"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b4"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("a4"), Piece::ATTACKER).unwrap();

        let res = bfs(
            |sq| board.board[sq] == Piece::EMPTY,
            &PRECOMPUTED.vertical_horizontal_neighbors,
            &vec![
                get_square_from_algebraic("a1"),
                get_square_from_algebraic("k1"),
            ],
        );

        expect_res_size(&res, SQS - 7);
    }

    #[test]
    fn correct_handle_accessible_fields() {
        let mut board = Board::new();

        board.set_piece(get_square_from_algebraic("a10"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b11"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("c11"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("d10"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("c9"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b9"), Piece::ATTACKER).unwrap();

        board.set_piece(get_square_from_algebraic("b10"), Piece::KING).unwrap();
        println!("{:?}", board);

        let res = bfs(
            |sq| board.board[sq] != Piece::ATTACKER,
            &PRECOMPUTED.vertical_horizontal_neighbors,
            &vec![board.king_sq as Square],
        );

        expect_res_size(&res, 2);
    }
}