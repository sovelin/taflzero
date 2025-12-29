use crate::board::types::Side;

mod king_is_surrounded;
mod defenders_is_surrounded;
mod check_fort;
pub mod terminal;
mod evaluate;

pub use evaluate::evaluate;

pub static MATE_SCORE: i32 = 10000;
static MATE_SCORE_THRESHOLD: i32 = MATE_SCORE - 200;

pub fn is_mate_score(score: i32) -> bool {
    score.abs() >= MATE_SCORE_THRESHOLD
}

pub fn sided_evaluation(score: i32, side: Side) -> i32 {
    match side {
        Side::DEFENDERS => score,
        Side::ATTACKERS => -score,
    }
}

pub use evaluate::evaluate_king_mobility;