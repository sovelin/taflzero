#[cfg(test)]
mod tests {
    use std::error::Error;

    use crate::board::Board;
    use crate::board::constants::SQS;
    use crate::board::rules::RulesEnum;
    use crate::board::types::Piece;
    use crate::board::utils::get_square_from_algebraic;
    use crate::tests::*;

    #[test]
    fn set_piece_places_attacker_correctly() -> Result<(), Box<dyn Error>> {
        let mut b = Board::new();
        b.set_piece(get_square_from_algebraic("f3"), Piece::ATTACKER)?;

        expect_attacker_on(&b, "f3");
        expect_attackers_count(&b, 1);

        Ok(())
    }

    #[test]
    fn set_2_attackers_and_defenders_correctly() -> Result<(), Box<dyn Error>> {
        let mut b = Board::new();
        b.set_piece(get_square_from_algebraic("f3"), Piece::ATTACKER)?;
        b.set_piece(get_square_from_algebraic("c6"), Piece::ATTACKER)?;
        b.set_piece(get_square_from_algebraic("i9"), Piece::DEFENDER)?;

        expect_attacker_on(&b, "f3");
        expect_attacker_on(&b, "c6");
        expect_attackers_count(&b, 2);

        expect_defender_on(&b, "i9");
        expect_defenders_count(&b, 1);

        Ok(())
    }

    #[test]
    fn set_king() -> Result<(), Box<dyn Error>> {
        let mut b = Board::new();
        b.set_piece(get_square_from_algebraic("f6"), Piece::KING)?;
        expect_king_on(&b, "f6");

        Ok(())
    }

    #[test]
    fn clear_resets_everything() -> Result<(), Box<dyn Error>> {
        let mut b = Board::new();

        b.set_piece(12, Piece::ATTACKER)?;
        b.set_piece(40, Piece::DEFENDER)?;
        b.set_piece(50, Piece::KING)?;

        b.clear();

        expect_defenders_count(&b, 0);
        expect_attackers_count(&b, 0);
        expect_king_not_exists(&b);
        assert_eq!(b.board.iter().all(|p| *p == Piece::EMPTY), true);

        Ok(())
    }

    #[test]
    fn clear_piece_correctly() -> Result<(), Box<dyn Error>> {
        let mut b = Board::new();

        b.set_piece(get_square_from_algebraic("c1"), Piece::ATTACKER)?;
        b.set_piece(get_square_from_algebraic("c2"), Piece::DEFENDER)?;
        b.clear_piece(get_square_from_algebraic("c1"));

        expect_no_pice_on(&b, "c1");
        // expect_defender_on(&b, "c2");
        // expect_attackers_count(&b, 0);
        // expect_defenders_count(&b, 1);

        Ok(())
    }

    // Both occupancy vectors are indexed by row/column, so each must be exactly as long
    // as the side of the board the variant asks for.
    #[test]
    fn set_rules_resizes_both_occupancy_vectors() {
        for variant in [
            RulesEnum::Copenhagen11x11,
            RulesEnum::Historical11x11,
            RulesEnum::Tablut9x9,
        ] {
            let mut board = Board::new();
            board.set_rules(variant);

            let board_size = board.board_size();

            assert_eq!(
                board.row_occ.len(),
                board_size,
                "row_occ length on a {board_size}x{board_size} board"
            );
            assert_eq!(
                board.col_occ.len(),
                board_size,
                "col_occ length on a {board_size}x{board_size} board"
            );
        }
    }

    // Switching back to a bigger board must leave both vectors long enough to index.
    #[test]
    fn set_rules_grows_occupancy_when_switching_back_to_11x11() {
        let mut board = Board::new();
        board.set_rules(RulesEnum::Tablut9x9);
        board.set_rules(RulesEnum::Copenhagen11x11);

        assert_eq!(board.row_occ.len(), 11);
        assert_eq!(board.col_occ.len(), 11);
    }

    const ALL_VARIANTS: [RulesEnum; 3] = [
        RulesEnum::Copenhagen11x11,
        RulesEnum::Historical11x11,
        RulesEnum::Tablut9x9,
    ];

    // The piece buffer handed out by `Board::board()` is what `EngineClient::get_board_state`
    // forwards to the UI, so it must describe exactly the squares the variant has — no
    // phantom cells past the end of a smaller board.
    #[test]
    fn exported_board_state_matches_variant_square_count() {
        for variant in ALL_VARIANTS {
            let mut board = Board::new();
            board.set_rules(variant);

            let board_size = board.board_size();
            let sqs = board_size * board_size;

            assert_eq!(
                board.board().len(),
                sqs,
                "board state exported for a {board_size}x{board_size} variant"
            );
        }
    }

    // Guard, not a bug report: the fixed-size arrays inside `Board` are sized by the SQS
    // constant. They are wide enough for every variant that exists today, and this test
    // fires the moment a variant larger than 11x11 is added.
    #[test]
    fn fixed_size_arrays_cover_every_variant() {
        for variant in ALL_VARIANTS {
            let board_size = variant.rules().board_size;
            let sqs = board_size * board_size;

            assert!(
                sqs <= SQS,
                "{board_size}x{board_size} needs {sqs} squares but Board is fixed at {SQS}"
            );
        }
    }
}
