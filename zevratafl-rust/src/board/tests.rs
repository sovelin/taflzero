#[cfg(test)]
mod tests {
    use crate::board::board::Board;
    use crate::board::types::{Piece};
    use crate::board::utils::{get_square_from_algebraic};
    use crate::tests::*;

    #[test]
    fn set_piece_places_attacker_correctly() {
        let mut b = Board::new();
        b.set_piece(get_square_from_algebraic("f3"), Piece::ATTACKER);

        expect_attacker_on(&b, "f3");
        expect_attackers_count(&b, 1);
    }

    #[test]
    fn set_2_attackers_and_defenders_correctly() {
        let mut b = Board::new();
        b.set_piece(get_square_from_algebraic("f3"), Piece::ATTACKER);
        b.set_piece(get_square_from_algebraic("c6"), Piece::ATTACKER);
        b.set_piece(get_square_from_algebraic("i9"), Piece::DEFENDER);

        expect_attacker_on(&b, "f3");
        expect_attacker_on(&b, "c6");
        expect_attackers_count(&b, 2);

        expect_defender_on(&b, "i9");
        expect_defenders_count(&b, 1);
    }

    #[test]
    fn set_king() {
        let mut b = Board::new();
        b.set_piece(get_square_from_algebraic("f6"), Piece::KING);
        expect_king_on(&b, "f6");
    }

    #[test]
    fn clear_resets_everything() {
        let mut b = Board::new();

        b.set_piece(12, Piece::ATTACKER);
        b.set_piece(40, Piece::DEFENDER);
        b.set_piece(50, Piece::KING);

        b.clear();

        expect_defenders_count(&b, 0);
        expect_attackers_count(&b, 0);
        expect_king_not_exists(&b);
        assert_eq!(b.board.iter().all(|p| *p == Piece::EMPTY), true);
    }

    #[test]
    fn clear_piece_correctly() {
        let mut b = Board::new();

        b.set_piece(get_square_from_algebraic("c1"), Piece::ATTACKER);
        b.set_piece(get_square_from_algebraic("c2"), Piece::DEFENDER);
        b.clear_piece(get_square_from_algebraic("c1"));

        expect_no_pice_on(&b, "c1");
        // expect_defender_on(&b, "c2");
        // expect_attackers_count(&b, 0);
        // expect_defenders_count(&b, 1);
    }
}
