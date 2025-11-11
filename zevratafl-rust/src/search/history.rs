use crate::board::constants::SQS;
use crate::board::types::{Side, Square};
use crate::moves::mv::Move;
use crate::search::constants::MAX_HISTORY_SCORE;

pub struct History {
    data: [[[i32; SQS]; SQS]; 2],
}

impl History {
    #[inline]
    pub fn new() -> Self {
        Self {
            data: [[[0; SQS]; SQS]; 2],
        }
    }

    #[inline]
    pub fn update(&mut self, side: Side, mv: Move, depth: i32) {

        let entry = &mut self.data[side as usize][mv.from()][mv.to()];
        *entry += depth * depth;

        if *entry > MAX_HISTORY_SCORE {
            *entry >>= 1;
        }
    }

    #[inline]
    pub fn get(&self, side: Side, from: Square, to: Square) -> i32 {
        self.data[side as usize][from][to]
    }
}
