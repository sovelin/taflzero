use crate::board::{Board, PRECOMPUTED};
use crate::board::types::{Side, Square};
use crate::evaluation::check_fort::check_fort;
use crate::evaluation::defenders_is_surrounded::defenders_is_surrounded;
use crate::evaluation::king_is_surrounded::king_is_surrounded;

pub fn is_threefold_repetition(board: &Board) -> bool {
    if let Some(value) = board.rep_table.get(&board.zobrist) {
        *value >= 3
    } else {
        false
    }
}

pub fn check_terminal(board: &mut Board) -> Option<Side> {
    if is_threefold_repetition(board) {
        return Some(Side::ATTACKERS);
    }

    if PRECOMPUTED.corners_sq.contains(&(board.king_sq as Square)) {
        return Some(Side::DEFENDERS);
    }

    if king_is_surrounded(board) {
        return Some(Side::ATTACKERS);
    }

    if defenders_is_surrounded(board) {
        return Some(Side::ATTACKERS);
    }

    if check_fort(board) {
        return Some(Side::DEFENDERS);
    }

    None
}

#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::board::types::{Piece, Side};
    use crate::board::utils::get_square_from_algebraic;
    use crate::evaluation::terminal::check_terminal;
    use crate::moves::mv::create_move_from_algebraic;

    #[test]
    fn three_repetition_is_attacker_win() {
        let mut board = Board::new();
        board.setup_initial_position().unwrap();

        board.make_move_simple(create_move_from_algebraic("a4b4").unwrap()).unwrap();
        board.make_move_simple(create_move_from_algebraic("e5d5").unwrap()).unwrap();
        board.make_move_simple(create_move_from_algebraic("b4a4").unwrap()).unwrap();
        board.make_move_simple(create_move_from_algebraic("d5e5").unwrap()).unwrap();

        board.make_move_simple(create_move_from_algebraic("a4b4").unwrap()).unwrap();
        board.make_move_simple(create_move_from_algebraic("e5d5").unwrap()).unwrap();
        board.make_move_simple(create_move_from_algebraic("b4a4").unwrap()).unwrap();
        board.make_move_simple(create_move_from_algebraic("d5e5").unwrap()).unwrap();

        board.make_move_simple(create_move_from_algebraic("a4b4").unwrap()).unwrap();
        board.make_move_simple(create_move_from_algebraic("e5d5").unwrap()).unwrap();
        board.make_move_simple(create_move_from_algebraic("b4a4").unwrap()).unwrap();
        board.make_move_simple(create_move_from_algebraic("d5e5").unwrap()).unwrap();

        let result = check_terminal(&mut board);
        assert_eq!(result, Some(Side::DEFENDERS));
    }

    #[test]
    fn two_repetition_is_not_terminate_win() {
        let mut board = Board::new();
        board.setup_initial_position().unwrap();

        board.make_move_simple(create_move_from_algebraic("a4b4").unwrap()).unwrap();
        board.make_move_simple(create_move_from_algebraic("e5d5").unwrap()).unwrap();
        board.make_move_simple(create_move_from_algebraic("b4a4").unwrap()).unwrap();
        board.make_move_simple(create_move_from_algebraic("d5e5").unwrap()).unwrap();

        board.make_move_simple(create_move_from_algebraic("a4b4").unwrap()).unwrap();
        board.make_move_simple(create_move_from_algebraic("e5d5").unwrap()).unwrap();
        board.make_move_simple(create_move_from_algebraic("b4a4").unwrap()).unwrap();
        board.make_move_simple(create_move_from_algebraic("d5e5").unwrap()).unwrap();

        let result = check_terminal(&mut board);
        assert_eq!(result, None);
    }

    #[test]
    fn king_on_corner_is_defender_win() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("a1"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("b2"), Piece::DEFENDER).unwrap();
        board.set_side(Side::DEFENDERS);


        let result = check_terminal(&mut board);
        assert_eq!(result, Some(Side::DEFENDERS));
    }

    #[test]
    fn king_on_a11_win() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("a11"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("b2"), Piece::DEFENDER).unwrap();
        board.set_side(Side::DEFENDERS);


        let result = check_terminal(&mut board);
        assert_eq!(result, Some(Side::DEFENDERS));
    }

    #[test]
    fn king_make_move() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("a2"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("b2"), Piece::DEFENDER).unwrap();
        board.set_side(Side::DEFENDERS);

        board.make_move_simple(create_move_from_algebraic("a2a1").unwrap()).unwrap();


        let result = check_terminal(&mut board);
        assert_eq!(result, Some(Side::DEFENDERS));
    }
}