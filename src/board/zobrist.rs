use crate::board::constants::SQS;
use crate::board::types::ZobristHash;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use std::sync::LazyLock;

const PIECES_COUNT: usize = 4;

type ZobristTable = [[ZobristHash; SQS]; PIECES_COUNT];

pub struct ZobristData {
    pub table: ZobristTable,
    pub side: ZobristHash,
}

fn generate_zobrist() -> ZobristData {
    let mut rng = StdRng::seed_from_u64(12345);

    let mut table = [[0; SQS]; PIECES_COUNT];

    for p in 0..PIECES_COUNT {
        for sq in 0..SQS {
            table[p][sq] = rng.next_u64()
        }
    }

    let side = rng.next_u64();

    ZobristData { table, side }
}

pub static ZOBRIST_DATA: LazyLock<ZobristData> = LazyLock::new(generate_zobrist);
