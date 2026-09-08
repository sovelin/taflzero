pub type Mask = u16;

#[inline]
fn build_line_mask(pos: usize, occ: Mask, board_size: usize) -> Mask {
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
    while i < board_size {
        let bit = 1u16 << i;
        if (occ & bit) != 0 {
            break;
        }
        mask |= bit;
        i += 1;
    }

    mask
}

pub fn create_line_moves(board_size: usize) -> Vec<Vec<Mask>> {
    let occ_states = 1usize << board_size;

    (0..board_size)
        .map(|pos| {
            (0..occ_states)
                .map(|occ| build_line_mask(pos, occ as Mask, board_size))
                .collect()
        })
        .collect()
}
