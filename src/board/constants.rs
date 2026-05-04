use crate::board::types::OptionalSquare;

pub const BOARD_SIZE: usize = 11;
pub const SQS: usize = BOARD_SIZE * BOARD_SIZE;
pub const HOLE: OptionalSquare = -1;
pub const ATTACKERS_MAX: usize = SQS;
pub const DEFENDERS_MAX: usize = SQS;

pub const INITIAL_FEN: &str =
    "3aaaaa3/5a5/11/a4d4a/a3ddd3a/aa1ddkdd1aa/a3ddd3a/a4d4a/11/5a5/3aaaaa3 a";
