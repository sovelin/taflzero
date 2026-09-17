#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::board::constants::INITIAL_FEN;
    use crate::board::types::{Piece, Side};

    mod set_fen {
        use super::*;
        use crate::tests::{
            expect_attacker_on, expect_attackers_count, expect_defender_on, expect_defenders_count,
            expect_king_not_exists, expect_king_on, expect_side_to_be,
        };

        #[test]
        fn one_king_two_attackers_two_defenders_defenders_to_move() {
            let fen = "11/4a6/11/11/7d3/5k5/11/1d9/8a2/11/10a d";

            let mut b = Board::new();
            b.set_fen(fen).expect("set_fen failed");

            expect_attackers_count(&b, 3);
            expect_defenders_count(&b, 2);

            expect_attacker_on(&b, "i3");
            expect_attacker_on(&b, "e10");
            expect_attacker_on(&b, "k1");

            expect_king_on(&b, "f6");

            expect_defender_on(&b, "b4");
            expect_defender_on(&b, "h7");
            assert_eq!(b.side_to_move, Side::DEFENDERS);
        }

        #[test]
        fn attackers_to_move() {
            let fen = "11/4a6/11/11/7d3/5k5/11/1d9/8a2/11/11 a";

            let mut b = Board::new();
            b.set_fen(fen).expect("set_fen failed");

            expect_side_to_be(&b, Side::ATTACKERS);
        }

        #[test]
        fn set_twice() {
            let mut b = Board::new();
            b.set_fen("11/4a6/11/11/7d3/5k5/11/1d9/8a2/11/11 a")
                .expect("set_fen #1 failed");

            b.set_fen("11/11/11/11/11/11/11/11/11/11/11 a")
                .expect("set_fen #2 failed");

            expect_side_to_be(&b, Side::ATTACKERS);
            expect_attackers_count(&b, 0);
            expect_defenders_count(&b, 0);
            expect_king_not_exists(&b);
        }
    }

    mod get_fen {
        use std::error::Error;

        use super::*;

        #[test]
        fn one_king_two_attackers_two_defenders_attackers_to_move() -> Result<(), Box<dyn Error>> {
            let mut b = Board::new();

            b.set_piece(b.get_square_from_algebraic("i3"), Piece::ATTACKER)?;
            b.set_piece(b.get_square_from_algebraic("f6"), Piece::KING)?;
            b.set_piece(b.get_square_from_algebraic("b4"), Piece::DEFENDER)?;
            b.set_piece(b.get_square_from_algebraic("h7"), Piece::DEFENDER)?;
            b.set_piece(b.get_square_from_algebraic("e10"), Piece::ATTACKER)?;
            b.set_piece(b.get_square_from_algebraic("k1"), Piece::ATTACKER)?;

            b.side_to_move = Side::ATTACKERS;

            let fen = b.get_fen();
            assert_eq!(fen, "11/4a6/11/11/7d3/5k5/11/1d9/8a2/11/10a - a 0 0");

            Ok(())
        }
    }

    #[test]
    fn zobrists_should_be_same() {
        let mut b = Board::new();
        b.set_fen(INITIAL_FEN).expect("set_fen failed");

        let fen0 = b.get_fen();
        let z0 = b.zobrist;

        b.set_fen(&fen0).expect("set_fen failed");
        let z1 = b.zobrist;

        assert_eq!(z0, z1, "Zobrist should not change after set_fen(get_fen())");
    }

    #[test]
    fn roundtrip_set_and_get_fen_basic() {
        let fen = "11/4a6/11/11/7d3/5k5/11/1d9/8a2/11/10a f6 d 7 42";
        let mut b = Board::new();
        b.set_fen(fen).expect("set_fen failed");

        let back = b.get_fen();
        assert_eq!(back, fen);
    }

    // The short `<rows> <side>` layout predates the last-to square and the counters.
    // Game links, PGN exports and the curriculum buffer are full of it, so it has to
    // keep loading — filling the missing fields with their empty values.
    #[test]
    fn legacy_two_field_fen_still_loads() {
        let mut b = Board::new();
        b.set_fen("11/4a6/11/11/7d3/5k5/11/1d9/8a2/11/10a d")
            .expect("legacy set_fen failed");

        assert_eq!(b.side_to_move, Side::DEFENDERS);
        assert_eq!(b.last_move_to, crate::board::constants::HOLE);
        assert_eq!(b.halfmove_clock, 0);
        assert_eq!(b.halfmove_count, 0);
    }

    #[test]
    fn last_to_square_survives_a_roundtrip() {
        let mut b = Board::new();
        b.set_fen("11/4a6/11/11/7d3/5k5/11/1d9/8a2/11/10a i3 d 3 11")
            .expect("set_fen failed");

        assert_eq!(
            b.last_move_to,
            b.get_square_from_algebraic("i3") as crate::board::types::OptionalSquare
        );
        assert!(b.get_fen().contains(" i3 d 3 11"));
    }
}
