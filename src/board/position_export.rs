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

        for row in 0..11 {
            for col in 0..11 {
                let idx = row * 11 + col;
                let byte = idx / 8;
                let bit = idx % 8;

                match board.board[get_square(row, col)] {
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

        for idx in 0..121usize {
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
    use crate::board::types::Side;
    use crate::board::utils::get_square;
    use crate::board::{Board, set_board_from_str};

    fn defenders_plane(bit_position: &BitPosition, plane: usize) -> String {
        let base = plane * 16;
        let mut out = String::new();

        for i in 0..BOARD_SIZE * BOARD_SIZE {
            let row = BOARD_SIZE - 1 - i / BOARD_SIZE;
            let col = i % BOARD_SIZE;
            let sq = get_square(row, col);

            let byte = sq / 8;
            let bit = sq % 8;
            let v = (bit_position.planes[base + byte] >> bit) & 1;

            out.push(if v == 1 { 'X' } else { '.' });

            if (sq + 1) % 11 == 0 {
                out.push('\n');
            }
        }

        out
    }

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
}
