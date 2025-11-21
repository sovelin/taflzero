use crate::board::constants::HOLE;
use crate::board::types::{Piece, Side, Square};
use crate::board::utils::get_square;
use crate::board::{Board, PRECOMPUTED};
use crate::moves::masks::LINE_MOVES;
use crate::moves::mv::Move;

pub static MAX_MOVES: usize = 1024;

static THRONE_MASK: u16 = 1 << 5;
static BOUNDARY_MASK: u16 = 1 | (1 << 10);

pub struct MoveGen {
    pub moves: [Move; MAX_MOVES],
    pub move_scores: [i32; MAX_MOVES],
    pub count: usize,
}

impl MoveGen {
    pub fn new() -> Self {
        MoveGen {
            moves: [Move::default(); MAX_MOVES],
            count: 0,
            move_scores: [0; MAX_MOVES],
        }
    }

    fn add_move(&mut self, mv: Move) {
        if self.count < MAX_MOVES {
            self.moves[self.count] = mv;
            self.count += 1;
        } else {
            panic!("Exceeded maximum number of moves in MoveGen");
        }
    }

    fn disable_blocked_squares(&mut self, beam: u16, piece: Piece, row_or_col_index: usize) -> u16 {
        let mut beam = beam;
        if piece == Piece::ATTACKER || piece == Piece::DEFENDER {
            if row_or_col_index == 5 {
                beam &= !THRONE_MASK;
            } else if row_or_col_index == 0 || row_or_col_index == 10 {
                beam &= !BOUNDARY_MASK;
            }
        }

        beam
    }

    fn generate_horizontal_moves(&mut self, board: &Board, from: usize) {
        let row = PRECOMPUTED.row[from];
        let col = PRECOMPUTED.col[from];
        let row_occ = board.row_occ[row];

        let horizontal = LINE_MOVES[col][row_occ as usize] & !(1 << col);
        let mut horizontal_moves = self.disable_blocked_squares(horizontal, board.board[from], row);

        let mut cur = 1;

        while horizontal_moves != 0 {
            if cur & horizontal_moves != 0 {
                let to = get_square(row, cur.trailing_zeros() as usize);

                self.add_move(Move::new(from, to));
            }
            horizontal_moves &= !cur;
            cur <<= 1;
        }
    }

    fn generate_vertical_moves(&mut self, board: &Board, from: usize) {
        let col = PRECOMPUTED.col[from];
        let row = PRECOMPUTED.row[from];
        let col_occ = board.col_occ[col];

        let vertical = LINE_MOVES[row][col_occ as usize] & !(1 << row);
        let mut vertical_moves = self.disable_blocked_squares(vertical, board.board[from], col);

        let mut cur = 1;

        while vertical_moves != 0 {
            if cur & vertical_moves != 0 {
                let to = get_square(cur.trailing_zeros() as usize, col);
                self.add_move(Move::new(from, to));
            }
            vertical_moves &= !cur;
            cur <<= 1;
        }
    }

    fn generate_moves_for_piece(&mut self, board: &Board, from: Square) {
        self.generate_horizontal_moves(board, from);
        self.generate_vertical_moves(board, from);
    }

    fn generate_defender_moves(&mut self, board: &Board) {
        for i in 0..board.defenders_count {
            let sq = board.defenders[i as usize];
            self.generate_moves_for_piece(board, sq);
        }

        if board.king_sq != HOLE {
            self.generate_moves_for_piece(board, board.king_sq as Square);
        }
    }

    fn generate_attacker_moves(&mut self, board: &Board) {
        for i in 0..board.attackers_count {
            let sq = board.attackers[i as usize];
            self.generate_moves_for_piece(board, sq);
        }
    }

    pub fn generate_moves(&mut self, board: &Board) {
        self.count = 0;

        if board.side_to_move == Side::ATTACKERS {
            self.generate_attacker_moves(board);
        } else {
            self.generate_defender_moves(board);
        }
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn moves(&self) -> &[Move] {
        &self.moves[..self.count]
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::board::Board;
    use crate::board::types::{Piece, Side};
    use crate::board::utils::get_square_from_algebraic;
    use crate::moves::mv::create_move_from_algebraic;

    fn expect_moves_exists(move_gen: &MoveGen, moves: Vec<&str>) {
        for mv_str in moves {
            let mv = create_move_from_algebraic(mv_str).unwrap();
            assert!(
                move_gen.moves[..move_gen.count].contains(&mv),
                "Expected move {} to exist",
                mv_str
            );
        }
    }

    fn expect_moves_not_exists(move_gen: &MoveGen, moves: Vec<&str>) {
        for mv_str in moves {
            let mv = create_move_from_algebraic(mv_str).unwrap();
            assert!(
                !move_gen.moves[..move_gen.count].contains(&mv),
                "Expected move {} to NOT exist",
                mv_str
            );
        }
    }

    fn expect_moves_count(move_gen: &MoveGen, expected_count: usize) {
        assert_eq!(
            move_gen.count, expected_count,
            "Expected move count to be {}, but got {}",
            expected_count, move_gen.count
        );
    }

    #[test]
    fn empty_position_zero_moves() {
        let board = Board::new();
        let mut movegen = MoveGen::new();
        movegen.generate_moves(&board);

        expect_moves_count(&movegen, 0);
    }

    #[test]
    fn one_piece_on_b1() {
        let mut board = Board::new();
        board.side_to_move = Side::ATTACKERS;
        board.set_piece(get_square_from_algebraic("b1"), Piece::ATTACKER);

        let mut movegen = MoveGen::new();
        movegen.generate_moves(&board);

        expect_moves_count(&movegen, 18);

        expect_moves_exists(
            &movegen,
            vec![
                "b1c1", "b1d1", "b1e1", "b1f1", "b1g1", "b1h1", "b1i1", "b1j1", "b1b2", "b1b3",
                "b1b4", "b1b5", "b1b6", "b1b7", "b1b8", "b1b9", "b1b10", "b1b11",
            ],
        );

        expect_moves_not_exists(&movegen, vec!["b1a1", "b1a10"]);
    }

    #[test]
    fn one_piece_on_b1_and_enemy_piece_on_c1() {
        let mut board = Board::new();
        board.side_to_move = Side::ATTACKERS;
        board.set_piece(get_square_from_algebraic("b1"), Piece::ATTACKER);
        board.set_piece(get_square_from_algebraic("c1"), Piece::DEFENDER);

        let mut movegen = MoveGen::new();
        movegen.generate_moves(&board);

        expect_moves_count(&movegen, 10);

        expect_moves_exists(
            &movegen,
            vec![
                "b1b2", "b1b3", "b1b4", "b1b5", "b1b6", "b1b7", "b1b8", "b1b9", "b1b10", "b1b11",
            ],
        );
    }

    #[test]
    fn one_piece_on_a2_and_enemy_piece_on_a3() {
        let mut board = Board::new();
        board.side_to_move = Side::ATTACKERS;
        board.set_piece(get_square_from_algebraic("a2"), Piece::ATTACKER);
        board.set_piece(get_square_from_algebraic("a3"), Piece::DEFENDER);

        let mut movegen = MoveGen::new();
        movegen.generate_moves(&board);

        expect_moves_count(&movegen, 10);

        expect_moves_exists(
            &movegen,
            vec![
                "a2b2", "a2c2", "a2d2", "a2e2", "a2f2", "a2g2", "a2h2", "a2i2", "a2j2", "a2k2",
            ],
        );
    }

    #[test]
    fn piece_should_not_move_onto_throne_f6() {
        let mut board = Board::new();
        board.side_to_move = Side::DEFENDERS;
        board.set_piece(get_square_from_algebraic("f5"), Piece::DEFENDER);

        let mut movegen = MoveGen::new();
        movegen.generate_moves(&board);

        expect_moves_not_exists(&movegen, vec!["f5f6"]);
    }

    #[test]
    fn king_could_move_onto_throne_f6() {
        let mut board = Board::new();
        board.side_to_move = Side::DEFENDERS;
        board.set_piece(get_square_from_algebraic("f5"), Piece::KING);

        let mut movegen = MoveGen::new();
        movegen.generate_moves(&board);

        expect_moves_exists(&movegen, vec!["f5f6"]);
    }

    #[test]
    fn king_could_move_onto_corner_a1() {
        let mut board = Board::new();
        board.side_to_move = Side::DEFENDERS;
        board.set_piece(get_square_from_algebraic("a2"), Piece::KING);

        let mut movegen = MoveGen::new();
        movegen.generate_moves(&board);

        expect_moves_count(&movegen, 20);
        expect_moves_exists(&movegen, vec!["a2a1"]);
    }

    #[test]
    fn king_have_zero_moves_in_blockade() {
        let mut board = Board::new();
        board.side_to_move = Side::DEFENDERS;

        board.set_piece(get_square_from_algebraic("e5"), Piece::KING);
        board.set_piece(get_square_from_algebraic("e4"), Piece::ATTACKER);
        board.set_piece(get_square_from_algebraic("e6"), Piece::ATTACKER);
        board.set_piece(get_square_from_algebraic("d5"), Piece::ATTACKER);
        board.set_piece(get_square_from_algebraic("f5"), Piece::ATTACKER);

        let mut movegen = MoveGen::new();
        movegen.generate_moves(&board);

        expect_moves_count(&movegen, 0);
    }
}
