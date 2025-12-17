use crate::{Board, POSSIBLE_MOVES_COUNT};
use crate::board::utils::{get_col, get_row};
use crate::mv::Move;

pub fn king_mobility_change(board: &Board, mv: Move) -> i32 {
    let row = get_row(board.king_sq as usize);
    let col = get_col(board.king_sq as usize);
    let hor_mobility = POSSIBLE_MOVES_COUNT[col][board.row_occ[row] as usize];
    let ver_mobility = POSSIBLE_MOVES_COUNT[row][board.col_occ[col] as usize];
    let old_total_mobility = hor_mobility + ver_mobility;

    let is_king_move = mv.from() == board.king_sq as usize;
    if is_king_move {
        let new_row = get_row(mv.to());
        let new_col = get_col(mv.to());

        let new_row_occ = board.row_occ[new_row] & !(1 << col); // remove king from row occupancy
        let new_col_occ = board.col_occ[new_col] & !(1 << row);

        let mut new_hor_mobility = POSSIBLE_MOVES_COUNT[new_col][new_row_occ as usize];
        let mut new_ver_mobility = POSSIBLE_MOVES_COUNT[new_row][new_col_occ as usize];

        new_hor_mobility &= !(1 << col);
        new_ver_mobility &= !(1 << row);

        let new_total_mobility = new_hor_mobility + new_ver_mobility;
        return (new_total_mobility as i32 - old_total_mobility as i32);
    } else {
        // compute if nothing changes
        let from_row = get_row(mv.from());
        let from_col = get_col(mv.from());
        let to_row = get_row(mv.to());
        let to_col = get_col(mv.to());

        if from_row == row {
            // horizontal move
            let mut new_row_occ = board.row_occ[row];
            new_row_occ &= !(1 << from_col); // remove piece from 'from' square
            new_row_occ |= 1 << to_col; // add piece to 'to' square

            let new_hor_mobility = POSSIBLE_MOVES_COUNT[col][new_row_occ as usize];
            let new_total_mobility = new_hor_mobility + ver_mobility;
            return (new_total_mobility as i32 - old_total_mobility as i32);
        } else if from_col == col {
            // vertical move
            let mut new_col_occ = board.col_occ[col];
            new_col_occ &= !(1 << from_row); // remove piece from 'from' square
            new_col_occ |= 1 << to_row; // add piece to 'to' square

            let new_ver_mobility = POSSIBLE_MOVES_COUNT[row][new_col_occ as usize];
            let new_total_mobility = hor_mobility + new_ver_mobility;
            return (new_total_mobility as i32 - old_total_mobility as i32);
        }
    }

    0
}

#[cfg(test)]
mod tests {
    use crate::Board;
    use crate::board::utils::get_square_from_algebraic;
    use crate::types::Piece;
    use super::*;

    #[test]
    fn test_king_vertical_mobility_change() {
        let mut board = Board::new();

        board.set_piece(get_square_from_algebraic("f6"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("f3"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("a6"), Piece::DEFENDER).unwrap();

        let mv = Move::new(get_square_from_algebraic("f3"), get_square_from_algebraic("f4"));

        let change = king_mobility_change(&board, mv);
        assert_eq!(change, -1);
    }

    #[test]
    fn test_king_horizontal_mobility_change_vertical() {
        let mut board = Board::new();

        board.set_piece(get_square_from_algebraic("f6"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("f3"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b6"), Piece::DEFENDER).unwrap();

        let mv = Move::new(get_square_from_algebraic("b6"), get_square_from_algebraic("a6"));

        let change = king_mobility_change(&board, mv);
        assert_eq!(change, 1);
    }

    #[test]
    fn test_king_not_affected_move() {
        let mut board = Board::new();

        board.set_piece(get_square_from_algebraic("f6"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("f3"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b6"), Piece::DEFENDER).unwrap();

        let mv = Move::new(get_square_from_algebraic("b7"), get_square_from_algebraic("a7"));

        let change = king_mobility_change(&board, mv);
        assert_eq!(change, 0);
    }

    #[test]
    fn test_king_move_horizontal_mobility_change() {
        let mut board = Board::new();

        board.set_piece(get_square_from_algebraic("f6"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("e6"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("g6"), Piece::DEFENDER).unwrap();
        // print board
        println!("{}", board);

        let mv = Move::new(get_square_from_algebraic("f6"), get_square_from_algebraic("f5"));

        let change = king_mobility_change(&board, mv);
        assert_eq!(change, 10);
    }
}