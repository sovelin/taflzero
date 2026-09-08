use std::sync::LazyLock;

pub const BOARD_SIZE: usize = 11;
pub const OCC_STATES: usize = 1 << BOARD_SIZE;

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Direction {
    Up = 0,
    Down = 1,
}

pub const ALL_DIRECTIONS: [Direction; 2] = [Direction::Up, Direction::Down];

pub type Mask = u16;

#[inline]
fn build_line_mask(pos: usize, occ: Mask) -> Mask {
    let mut mask: Mask = 0;

    let mut i = pos as isize - 1;
    while i >= 0 {
        let bit = 1u16 << i;
        if (occ & bit) != 0 {
            break;
        }
        mask |= bit;
        i -= 1;
    }

    let mut i = pos + 1;
    while i < BOARD_SIZE {
        let bit = 1u16 << i;
        if (occ & bit) != 0 {
            break;
        }
        mask |= bit;
        i += 1;
    }

    mask
}

pub static LINE_MOVES: LazyLock<[[Mask; OCC_STATES]; BOARD_SIZE]> = LazyLock::new(|| {
    let mut table = [[0u16; OCC_STATES]; BOARD_SIZE];
    for (pos, position) in table.iter_mut().enumerate().take(BOARD_SIZE) {
        for (occ, occupied) in position.iter_mut().enumerate().take(OCC_STATES) {
            *occupied = build_line_mask(pos, occ as Mask);
        }
    }
    table
});
