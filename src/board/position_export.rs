use crate::board::Board;
use crate::board::rules::RulesEnum;
use crate::board::types::{Piece, Side};
use crate::board::utils::get_square;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BitPosition {
    pub planes: [u8; 16 * 3], // [attackers | defenders | king]
    pub stm: u8,              // 1 for defenders, 0 for attackers
    pub rep: u8,              // repetition count for this position in the current game
}

impl BitPosition {
    pub fn from_board(board: &Board, rep: u8) -> Self {
        let mut planes = [0u8; 16 * 3];
        let board_size = board.board_size();

        for row in 0..board_size {
            for col in 0..board_size {
                let idx = row * board_size + col;
                let byte = idx / 8;
                let bit = idx % 8;

                match board.board[get_square(row, col, board_size)] {
                    Piece::ATTACKER => {
                        planes[byte] |= 1 << bit;
                    }
                    Piece::DEFENDER => {
                        planes[16 + byte] |= 1 << bit;
                    }
                    Piece::KING => {
                        planes[32 + byte] |= 1 << bit;
                    }
                    Piece::EMPTY => {}
                }
            }
        }

        Self {
            planes,
            stm: board.side_to_move as u8,
            rep,
        }
    }

    pub fn to_board(&self, variant: RulesEnum) -> Board {
        let mut board = Board::new();
        board.set_rules(variant);

        let board_size = board.board_size();
        let sqs = board_size * board_size;

        for idx in 0..sqs {
            let byte = idx / 8;
            let bit = idx % 8;

            if (self.planes[byte] >> bit) & 1 == 1 {
                board.set_piece(idx, Piece::ATTACKER).ok();
            } else if (self.planes[16 + byte] >> bit) & 1 == 1 {
                board.set_piece(idx, Piece::DEFENDER).ok();
            } else if (self.planes[32 + byte] >> bit) & 1 == 1 {
                board.set_piece(idx, Piece::KING).ok();
            }
        }

        let side = if self.stm == 0 {
            Side::ATTACKERS
        } else {
            Side::DEFENDERS
        };
        board.set_side(side);
        board
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self as *const BitPosition) as *const u8,
                size_of::<BitPosition>(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::board::constants::BOARD_SIZE;
    use crate::board::position_export::BitPosition;
    use crate::board::rules::RulesEnum;
    use crate::board::types::{Piece, Side};
    use crate::board::utils::get_square;
    use crate::board::{Board, set_board_from_str};
    use indoc::indoc;

    fn plane_of_size(bit_position: &BitPosition, plane: usize, board_size: usize) -> String {
        let base = plane * 16;
        let mut out = String::new();

        for i in 0..board_size * board_size {
            let row = board_size - 1 - i / board_size;
            let col = i % board_size;
            let sq = get_square(row, col, board_size);

            let byte = sq / 8;
            let bit = sq % 8;
            let v = (bit_position.planes[base + byte] >> bit) & 1;

            out.push(if v == 1 { 'X' } else { '.' });

            if (sq + 1) % board_size == 0 {
                out.push('\n');
            }
        }

        out
    }

    fn defenders_plane(bit_position: &BitPosition, plane: usize) -> String {
        plane_of_size(bit_position, plane, BOARD_SIZE)
    }

    fn board_with(variant: RulesEnum, position: &str) -> Board {
        let mut board = Board::new();
        board.set_rules(variant);
        set_board_from_str(&mut board, position);
        board
    }

    fn tablut_board(position: &str) -> Board {
        board_with(RulesEnum::Tablut9x9, position)
    }

    /// Exporting to `BitPosition` and importing back must not lose anything the format
    /// claims to carry: piece placement, the king, piece counts and side to move.
    fn assert_round_trip(board: &Board, variant: RulesEnum) {
        let bit_position = BitPosition::from_board(board, 1);
        let restored = bit_position.to_board(variant);

        assert_eq!(restored.get_fen(), board.get_fen(), "fen");
        assert_eq!(restored.side_to_move, board.side_to_move, "side to move");
        assert_eq!(restored.king_sq, board.king_sq, "king square");
        assert_eq!(restored.attackers_count, board.attackers_count, "attackers");
        assert_eq!(restored.defenders_count, board.defenders_count, "defenders");
        assert_eq!(restored.zobrist, board.zobrist, "zobrist");

        let sqs = board.board_size() * board.board_size();
        for sq in 0..sqs {
            assert_eq!(restored.board[sq], board.board[sq], "square {sq}");
        }
    }

    const VARIANTS: [RulesEnum; 3] = [
        RulesEnum::Copenhagen11x11,
        RulesEnum::Historical11x11,
        RulesEnum::Tablut9x9,
    ];

    #[test]
    fn test_bit_position() {
        let mut board = Board::new();

        set_board_from_str(
            &mut board,
            "...........
             ...........
             ...........
             ...AAA......
             ..DAK.......
             ...AAA......
             ...........
             ...........
             ...........
             ...........
             ...........",
        );

        let bit_position = BitPosition::from_board(&board, 1);

        // Check the first plane (attackers)
        let attackers_plane = defenders_plane(&bit_position, 0);
        println!("Plane 0 (Attackers):\n{}", attackers_plane);
        assert_eq!(
            attackers_plane,
            r"...........
...........
...........
...XXX.....
...X.......
...XXX.....
...........
...........
...........
...........
...........
"
        );

        let plane1 = defenders_plane(&bit_position, 1);
        println!("Plane 1 (Defenders):\n{}", plane1);
        assert_eq!(
            plane1,
            r"...........
...........
...........
...........
..X........
...........
...........
...........
...........
...........
...........
"
        );

        let plane2 = defenders_plane(&bit_position, 2);
        println!("Plane 2 (King):\n{}", plane2);
        assert_eq!(
            plane2,
            r"...........
...........
...........
...........
....X......
...........
...........
...........
...........
...........
...........
"
        );
    }

    #[test]
    fn test_side_to_move() {
        let mut board = Board::new();
        assert_eq!(BitPosition::from_board(&board, 1).stm, 0); // Attackers to move

        board.side_to_move = Side::DEFENDERS;
        assert_eq!(BitPosition::from_board(&board, 1).stm, 1); // Defenders to move
    }

    #[test]
    fn tablut_9x9_planes() {
        let board = tablut_board(indoc! {"
            .........
            .........
            ....A....
            .........
            ...DK....
            .........
            ..A......
            .........
            .........
        "});

        let bit_position = BitPosition::from_board(&board, 1);

        assert_eq!(
            plane_of_size(&bit_position, 0, 9),
            indoc! {"
                .........
                .........
                ....X....
                .........
                .........
                .........
                ..X......
                .........
                .........
            "}
        );

        assert_eq!(
            plane_of_size(&bit_position, 1, 9),
            indoc! {"
                .........
                .........
                .........
                .........
                ...X.....
                .........
                .........
                .........
                .........
            "}
        );

        assert_eq!(
            plane_of_size(&bit_position, 2, 9),
            indoc! {"
                .........
                .........
                .........
                .........
                ....X....
                .........
                .........
                .........
                .........
            "}
        );
    }

    // A 9x9 square index must not leak into the 11x11 bit layout: on an 11x11 board the
    // king would land on a different bit than it does here.
    #[test]
    fn tablut_9x9_uses_9x9_square_indices() {
        let board = tablut_board(indoc! {"
            .........
            .........
            .........
            .........
            ....K....
            .........
            .........
            .........
            .........
        "});

        let bit_position = BitPosition::from_board(&board, 1);

        let king_sq = 4 * 9 + 4; // row 4, col 4 on a 9x9 board
        assert_eq!(
            (bit_position.planes[32 + king_sq / 8] >> (king_sq % 8)) & 1,
            1
        );
    }

    #[test]
    fn tablut_9x9_round_trip() {
        let mut board = Board::new();
        board.set_rules(RulesEnum::Tablut9x9);
        board.setup_initial_position().expect("initial position");

        let bit_position = BitPosition::from_board(&board, 1);
        let restored = bit_position.to_board(RulesEnum::Tablut9x9);

        assert_eq!(restored.get_fen(), board.get_fen());
    }

    #[test]
    fn tablut_9x9_side_to_move() {
        let mut board = tablut_board(indoc! {"
            .........
            .........
            .........
            .........
            ....K....
            .........
            .........
            .........
            .........
        "});

        assert_eq!(BitPosition::from_board(&board, 1).stm, 0);

        board.side_to_move = Side::DEFENDERS;
        assert_eq!(BitPosition::from_board(&board, 1).stm, 1);
    }

    #[test]
    fn round_trip_initial_position_every_variant() {
        for variant in VARIANTS {
            let mut board = Board::new();
            board.set_rules(variant);
            board.setup_initial_position().expect("initial position");

            assert_round_trip(&board, variant);
        }
    }

    #[test]
    fn round_trip_empty_board_every_variant() {
        for variant in VARIANTS {
            let mut board = Board::new();
            board.set_rules(variant);

            assert_eq!(board.king_sq, -1);
            assert_round_trip(&board, variant);
        }
    }

    // The strongest placement invariant: a lone king on each square in turn must come
    // back on exactly that square, for every board size.
    #[test]
    fn round_trip_king_on_every_square() {
        for variant in VARIANTS {
            let board_size = variant.rules().board_size;

            for sq in 0..board_size * board_size {
                let mut board = Board::new();
                board.set_rules(variant);
                board.set_piece(sq, Piece::KING).expect("set king");

                assert_round_trip(&board, variant);
            }
        }
    }

    #[test]
    fn round_trip_preserves_side_to_move() {
        for variant in VARIANTS {
            for side in [Side::ATTACKERS, Side::DEFENDERS] {
                let mut board = Board::new();
                board.set_rules(variant);
                board.set_piece(0, Piece::KING).expect("set king");
                board.set_side(side);

                assert_round_trip(&board, variant);
            }
        }
    }

    #[test]
    fn round_trip_11x11_position() {
        let board = board_with(
            RulesEnum::Copenhagen11x11,
            indoc! {"
                A.........A
                .D.......D.
                ..A.....A..
                ...D...D...
                ....A.A....
                .....K.....
                ....A.A....
                ...D...D...
                ..A.....A..
                .D.......D.
                A.........A
            "},
        );

        assert_eq!(board.attackers_count, 12);
        assert_eq!(board.defenders_count, 8);
        assert_round_trip(&board, RulesEnum::Copenhagen11x11);
    }

    #[test]
    fn round_trip_9x9_position() {
        let board = tablut_board(indoc! {"
            A.......A
            .D.....D.
            ..A...A..
            ...D.D...
            ....K....
            ...D.D...
            ..A...A..
            .D.....D.
            A.......A
        "});

        assert_eq!(board.attackers_count, 8);
        assert_eq!(board.defenders_count, 8);
        assert_round_trip(&board, RulesEnum::Tablut9x9);
    }

    // Corners and the throne are the squares most likely to be mishandled by an
    // index-mapping bug, and they differ between 11x11 and 9x9.
    #[test]
    fn round_trip_corners_and_throne() {
        for variant in VARIANTS {
            let board_size = variant.rules().board_size;
            let last = board_size - 1;

            let mut board = Board::new();
            board.set_rules(variant);

            board.set_piece(0, Piece::ATTACKER).expect("bottom left");
            board
                .set_piece(last, Piece::ATTACKER)
                .expect("bottom right");
            board
                .set_piece(last * board_size, Piece::ATTACKER)
                .expect("top left");
            board
                .set_piece(last * board_size + last, Piece::ATTACKER)
                .expect("top right");

            let center = (board_size - 1) / 2;
            board
                .set_piece(center * board_size + center, Piece::KING)
                .expect("throne");

            assert_round_trip(&board, variant);
        }
    }
}
