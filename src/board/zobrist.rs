use crate::board::types::ZobristHash;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const PIECES_COUNT: usize = 4;

type ZobristTable = Vec<Vec<ZobristHash>>;

pub struct ZobristData {
    pub table: ZobristTable,
    pub side: ZobristHash,
}

impl ZobristData {
    pub fn new(board_size: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(12345);
        let sqs = board_size * board_size;

        let mut table = vec![vec![0; sqs]; PIECES_COUNT];

        for p in table.iter_mut().take(PIECES_COUNT) {
            for sq in p.iter_mut().take(sqs) {
                *sq = rng.random::<u64>()
            }
        }

        let side = rng.random::<u64>();

        Self { table, side }
    }
}
