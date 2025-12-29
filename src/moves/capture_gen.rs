use crate::{Board, PRECOMPUTED};
use crate::board::utils::{get_col, get_row, get_square};
use crate::is_capture_possible::is_capture_possible;
use crate::masks::{Direction, LINE_CAPTURES};
use crate::movegen::MAX_MOVES;
use crate::mv::Move;
use crate::types::{Piece, Side, Square};

pub struct CaptureGen {
    pub captures: [Move; MAX_MOVES],
    pub count: usize,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Way {
    Left,
    Right,
    Top,
    Bottom,
}

impl CaptureGen {
    pub fn new() -> Self {
        CaptureGen {
            captures: [Move::default(); MAX_MOVES],
            count: 0,
        }
    }

    pub fn find_capture(&self, board: &Board, to: Square, way: Way) -> i8 {
        let row = get_row(to);
        let col = get_col(to);

        match way {
            Way::Left => {
                let direction = Direction::Down;
                let occ = board.row_occ[row];
                let new_col = LINE_CAPTURES[col][direction as usize][occ as usize];

                if new_col ==  -1 {
                    return -1;
                }

                get_square(row, new_col as Square) as i8
            },
            Way::Right => {
                let direction = Direction::Up;
                let occ = board.row_occ[row];
                let new_col = LINE_CAPTURES[col][direction as usize][occ as usize];

                if new_col ==  -1 {
                    return -1;
                }

                get_square(row, new_col as Square) as i8
            },
            Way::Top => {
                let direction = Direction::Up;
                let occ = board.col_occ[col];
                let new_row = LINE_CAPTURES[row][direction as usize][occ as usize];

                if new_row ==  -1 {
                    return -1;
                }

                get_square(new_row as Square, col) as i8
            },
            Way::Bottom => {
                let direction = Direction::Down;
                let occ = board.col_occ[col];
                let new_row = LINE_CAPTURES[row][direction as usize][occ as usize];

                if new_row ==  -1 {
                    return -1;
                }

                get_square(new_row as Square, col) as i8
            }
        }
    }

    fn add_capture(&mut self, mv: Move) {
        self.captures[self.count] = mv;
        self.count += 1;
    }

    fn get_square_on_way(&self, square: Square, way: Way) -> Option<Square> {
        match way {
            Way::Left => PRECOMPUTED.left_neighbor[square as usize],
            Way::Right => PRECOMPUTED.right_neighbor[square as usize],
            Way::Top => PRECOMPUTED.top_neighbor[square as usize],
            Way::Bottom => PRECOMPUTED.bottom_neighbor[square as usize],
        }
    }

    fn get_opposite_way(&self, way: Way) -> Way {
        match way {
            Way::Left => Way::Right,
            Way::Right => Way::Left,
            Way::Top => Way::Bottom,
            Way::Bottom => Way::Top,
        }
    }

    fn generate_attack_captures_for_way(&mut self, target: Square, board: &mut Board, way: Way, side: Side) {
        let to = self.get_square_on_way(target, way).unwrap();

        fn is_our_piece(piece: Piece, side: Side) -> bool {
            match side {
                Side::ATTACKERS => piece == Piece::ATTACKER,
                Side::DEFENDERS => piece == Piece::DEFENDER || piece == Piece::KING,
            }
        }

        let possible_ways = match way {
            Way::Left => vec![Way::Top, Way::Bottom, Way::Left],
            Way::Right => vec![Way::Top, Way::Bottom, Way::Right],
            Way::Top => vec![Way::Left, Way::Right, Way::Top],
            Way::Bottom => vec![Way::Left, Way::Right, Way::Bottom],
        };

        if possible_ways.contains(&Way::Bottom) {
            let bottom_capture = self.find_capture(board, to, Way::Bottom);

            if bottom_capture != -1 && is_our_piece(board.board[bottom_capture as Square], side) {
                self.add_capture(Move::new(bottom_capture as Square, to));
            }
        }

        if possible_ways.contains(&Way::Left) {
            let left_capture = self.find_capture(board, to, Way::Left);
            if left_capture != -1 && is_our_piece(board.board[left_capture as Square], side) {
                self.add_capture(Move::new(left_capture as Square, to));
            }
        }

        if possible_ways.contains(&Way::Top) {
            let top_capture = self.find_capture(board, to, Way::Top);
            if top_capture != -1 && is_our_piece(board.board[top_capture as Square], side) {
                self.add_capture(Move::new(top_capture as Square, to));
            }
        }

        if possible_ways.contains(&Way::Right) {
            let right_capture = self.find_capture(board, to, Way::Right);
            if right_capture != -1 && is_our_piece(board.board[right_capture as Square], side) {
                self.add_capture(Move::new(right_capture as Square, to));
            }
        }
    }

    fn is_capture_possible_in_way(&self, target: Square, way: Way, board: &mut Board, side: Side) -> bool {
        let mut res = false;
        let first_neighbor = self.get_square_on_way(target, way);
        let last_neighbor = self.get_square_on_way(target, self.get_opposite_way(way));
        let pice_to_set = match side {
            Side::ATTACKERS => Piece::ATTACKER,
            Side::DEFENDERS => Piece::DEFENDER,
        };

        if let Some(first) = first_neighbor {
            if let Some(last) = last_neighbor {
                // check if we can set left
                if board.board[first as Square] == Piece::EMPTY && !PRECOMPUTED.corners_sq.contains(&first) && first != PRECOMPUTED.throne_sq {
                    board.set_piece(first as Square, pice_to_set).unwrap();
                    res = is_capture_possible(board, target, first, last);
                    board.clear_piece(first as Square);
                }
            }
        }

        res
    }

    pub fn generate_captures(&mut self, board: &mut Board) {
        self.count = 0;
        let side = board.side_to_move;
        let enemy_pieces = if side == Side::ATTACKERS { board.defenders } else { board.attackers };

        for i in 0..enemy_pieces.len() as u8 {
            let target = enemy_pieces[i as usize];

            if self.is_capture_possible_in_way(target, Way::Right, board, side) {
                self.generate_attack_captures_for_way(target, board, Way::Right, side);
            }

            if self.is_capture_possible_in_way(target, Way::Left, board, side) {
                self.generate_attack_captures_for_way(target, board, Way::Left, side);
            }

            if self.is_capture_possible_in_way(target, Way::Top, board, side) {
                self.generate_attack_captures_for_way(target, board, Way::Top, side);
            }

            if self.is_capture_possible_in_way(target, Way::Bottom, board, side) {
                self.generate_attack_captures_for_way(target, board, Way::Bottom, side);
            }
        }
    }
    
    pub fn is_move_exists(&self, mv: Move) -> bool {
        for i in 0..self.count {
            if self.captures[i] == mv {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use crate::Board;
    use crate::board::utils::get_square_from_algebraic;
    use crate::moves::capture_gen::{CaptureGen, Way};
    use crate::mv::{create_move_from_algebraic, Move};
    use crate::types::{Piece, Side};

    fn expect_move_in_captures(capture_gen: &CaptureGen, mv_str: &str) -> bool {
        let mv = create_move_from_algebraic(mv_str).unwrap();
        for i in 0..capture_gen.count {
            if capture_gen.captures[i] == mv {
                return true;
            }
        }
        false
    }

    fn expect_move_count(capture_gen: &CaptureGen, expected_count: usize) {
        assert_eq!(capture_gen.count, expected_count);
    }

    #[test]
    fn test_capture_gen_bottom() {
        let mut board = Board::new();
        let mut capture_gen = CaptureGen::new();

        // set pieces on the board for testing
        board.set_piece(get_square_from_algebraic("e4"), Piece::DEFENDER);

        let target = get_square_from_algebraic("e8");

        let res = capture_gen.find_capture(&board, target, Way::Bottom);
        // assert something about res
        assert_eq!(res, get_square_from_algebraic("e4") as i8);
    }

    #[test]
    fn find_capture() {
        let board = Board::new();
        let mut capture_gen = CaptureGen::new();

        let target = get_square_from_algebraic("e8");

        let res = capture_gen.find_capture(&board, target, Way::Bottom);
        // assert something about res
        assert_eq!(res, -1);
    }


    mod attacker_captures {
        use super::*;

        mod left_captures {
            use super::*;

            #[test]
            fn from_top() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("f8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("d9"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen,1);
                expect_move_in_captures(&capture_gen, "d9d8");
            }

            #[test]
            fn from_bottom() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("f8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("d7"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen,1);
                expect_move_in_captures(&capture_gen, "d7d8");
            }

            #[test]
            fn from_left() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("f8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("c8"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen,1);
                expect_move_in_captures(&capture_gen, "c8d8");
            }

            #[test]
            fn from_right_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("f8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("k8"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen,0);
            }

            #[test]
            fn captures_for_occupied_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("f8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("c8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("d8"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen,0);
            }

            #[test]
            fn captures_with_our_piece_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("f8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("d7"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 0);
            }
        }

        mod right_captures {
            use super::*;

            #[test]
            fn from_top() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("f9"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("d8"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 1);
                expect_move_in_captures(&capture_gen, "f9f8");
            }

            #[test]
            fn from_bottom() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("f7"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("d8"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 1);
                expect_move_in_captures(&capture_gen, "f7f8");
            }

            #[test]
            fn from_right() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("g8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("d8"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 1);
                expect_move_in_captures(&capture_gen, "g8f8");
            }

            #[test]
            fn from_left_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("a8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("d8"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 0);
            }

            #[test]
            fn captures_for_occupied_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("f7"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("f8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("d8"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 0);
            }

            #[test]
            fn captures_with_our_piece_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("f7"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("d8"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 0);
            }
        }

        mod top_captures {
            use super::*;

            #[test]
            fn from_top() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("e10"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e7"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 1);
                expect_move_in_captures(&capture_gen, "e10e9");
            }

            #[test]
            fn from_bottom_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("e5"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e7"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 0);
            }

            #[test]
            fn from_right() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("f9"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e7"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 1);
                expect_move_in_captures(&capture_gen, "f9e9");
            }

            #[test]
            fn from_left() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("d9"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e7"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 1);
                expect_move_in_captures(&capture_gen, "d9e9");
            }

            #[test]
            fn captures_for_occupied_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("d9"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e7"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e9"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 0);
            }

            #[test]
            fn captures_with_our_piece_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("d9"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e7"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 0);
            }
        }

        mod bottom_captures {
            use super::*;

            #[test]
            fn from_bottom() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("e9"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e6"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 1);
                expect_move_in_captures(&capture_gen, "e6e7");
            }

            #[test]
            fn from_top_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("e9"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e10"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 0);
            }

            #[test]
            fn from_right() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("e9"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("f7"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 1);
                expect_move_in_captures(&capture_gen, "f7e7");
            }

            #[test]
            fn from_left() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("e9"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("b7"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 1);
                expect_move_in_captures(&capture_gen, "b7e7");
            }

            #[test]
            fn captures_for_occupied_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("e9"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("b7"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e7"), Piece::KING).unwrap(); // king for example

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 0);
            }

            #[test]
            fn captures_with_our_piece_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::ATTACKERS);
                board.set_piece(get_square_from_algebraic("e9"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("b7"), Piece::ATTACKER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 0);
            }
        }
    }

    mod defender_captures {
        use super::*;

        mod left_captures {
            use super::*;

            #[test]
            fn from_top() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("f8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("d9"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen,1);
                expect_move_in_captures(&capture_gen, "d9d8");
            }

            #[test]
            fn from_bottom() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("f8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("d7"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen,1);
                expect_move_in_captures(&capture_gen, "d7d8");
            }

            #[test]
            fn from_left() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("f8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("c8"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen,1);
                expect_move_in_captures(&capture_gen, "c8d8");
            }

            #[test]
            fn from_right_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("f8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("k8"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen,0);
            }

            #[test]
            fn captures_for_occupied_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("f8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("c8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("d8"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen,0);
            }

            #[test]
            fn captures_with_our_piece_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("f8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("d7"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 0);
            }
        }

        mod right_captures {
            use super::*;

            #[test]
            fn from_top() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("f9"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("d8"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 1);
                expect_move_in_captures(&capture_gen, "f9f8");
            }

            #[test]
            fn from_bottom() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("f7"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("d8"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 1);
                expect_move_in_captures(&capture_gen, "f7f8");
            }

            #[test]
            fn from_right() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("g8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("d8"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 1);
                expect_move_in_captures(&capture_gen, "g8f8");
            }

            #[test]
            fn from_left_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("a8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("d8"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 0);
            }

            #[test]
            fn captures_for_occupied_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("f7"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("f8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("d8"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 0);
            }

            #[test]
            fn captures_with_our_piece_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("f7"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("d8"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 0);
            }
        }

        mod top_captures {
            use super::*;

            #[test]
            fn from_top() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("e10"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e7"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 1);
                expect_move_in_captures(&capture_gen, "e10e9");
            }

            #[test]
            fn from_bottom_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("e5"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e7"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 0);
            }

            #[test]
            fn from_right() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("f9"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e7"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 1);
                expect_move_in_captures(&capture_gen, "f9e9");
            }

            #[test]
            fn from_left() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("d9"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e7"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 1);
                expect_move_in_captures(&capture_gen, "d9e9");
            }

            #[test]
            fn captures_for_occupied_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("d9"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e7"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e9"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 0);
            }

            #[test]
            fn captures_with_our_piece_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("d9"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e7"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 0);
            }
        }

        mod bottom_captures {
            use super::*;

            #[test]
            fn from_bottom() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("e9"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e6"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 1);
                expect_move_in_captures(&capture_gen, "e6e7");
            }

            #[test]
            fn from_top_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("e9"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("e10"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 0);
            }

            #[test]
            fn from_right() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("e9"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("f7"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 1);
                expect_move_in_captures(&capture_gen, "f7e7");
            }

            #[test]
            fn from_left() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("e9"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("b7"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 1);
                expect_move_in_captures(&capture_gen, "b7e7");
            }

            #[test]
            fn captures_for_occupied_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("e9"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
                board.set_piece(get_square_from_algebraic("b7"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e7"), Piece::KING).unwrap(); // king for example

                capture_gen.generate_captures(&mut board);

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 0);
            }

            #[test]
            fn captures_with_our_piece_impossible() {
                let mut board = Board::new();
                let mut capture_gen = CaptureGen::new();
                board.set_side(Side::DEFENDERS);
                board.set_piece(get_square_from_algebraic("e9"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER).unwrap();
                board.set_piece(get_square_from_algebraic("b7"), Piece::DEFENDER).unwrap();

                capture_gen.generate_captures(&mut board);
                expect_move_count(&capture_gen, 0);
            }
        }

        #[test]
        fn king_capture() {
            let mut board = Board::new();
            let mut capture_gen = CaptureGen::new();
            board.set_side(Side::DEFENDERS);
            board.set_piece(get_square_from_algebraic("e9"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("b7"), Piece::KING).unwrap();

            capture_gen.generate_captures(&mut board);
            expect_move_count(&capture_gen, 1);
            expect_move_in_captures(&capture_gen, "b7e7");
        }

        #[test]
        fn king_participant_in_capture() {
            let mut board = Board::new();
            let mut capture_gen = CaptureGen::new();
            board.set_side(Side::DEFENDERS);
            board.set_piece(get_square_from_algebraic("e9"), Piece::KING).unwrap();
            board.set_piece(get_square_from_algebraic("e8"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("b7"), Piece::DEFENDER).unwrap();

            capture_gen.generate_captures(&mut board);
            expect_move_count(&capture_gen, 1);
            expect_move_in_captures(&capture_gen, "b7e7");
        }
    }
}