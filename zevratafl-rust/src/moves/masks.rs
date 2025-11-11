use std::sync::LazyLock;

pub const BOARD_SIZE: usize = 11;
pub const OCC_STATES: usize = 1 << BOARD_SIZE;

pub type Mask = u16;

#[inline]
fn build_line_mask(pos: usize, occ: Mask) -> Mask {
    let mut mask: Mask = 0;

    let mut i = pos as isize - 1;
    while i >= 0 {
        let bit = 1u16 << i;
        if (occ & bit) != 0 { break; }
        mask |= bit;
        i -= 1;
    }

    let mut i = pos + 1;
    while i < BOARD_SIZE {
        let bit = 1u16 << i;
        if (occ & bit) != 0 { break; }
        mask |= bit;
        i += 1;
    }

    mask
}

pub static LINE_MOVES: LazyLock<[[Mask; OCC_STATES]; BOARD_SIZE]> = LazyLock::new(|| {
    let mut table = [[0u16; OCC_STATES]; BOARD_SIZE];
    for pos in 0..BOARD_SIZE {
        for occ in 0..OCC_STATES {
            table[pos][occ] = build_line_mask(pos, occ as Mask);
        }
    }
    table
});

pub static POSSIBLE_MOVES_COUNT: LazyLock<[[u8; OCC_STATES]; BOARD_SIZE]> = LazyLock::new(|| {
    let mut table = [[0u8; OCC_STATES]; BOARD_SIZE];
    for pos in 0..BOARD_SIZE {
        for occ in 0..OCC_STATES {
            table[pos][occ] = LINE_MOVES[pos][occ].count_ones() as u8;
        }
    }
    table
});