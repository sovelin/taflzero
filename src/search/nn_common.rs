use crate::board::PRECOMPUTED;
use crate::board::constants::SQS;
use crate::board::position_export::BitPosition;
use crate::utils::bfs::bfs;

pub const NUM_PLANES: usize = 11;
pub const POLICY_SIZE: usize = 4840;
pub const SAMPLE_SIZE: usize = NUM_PLANES * SQS;

pub struct NnOutput {
    pub policy: [f32; POLICY_SIZE],
    pub value: f32,
}

pub fn fill_input(input: &mut [f32], pos: &BitPosition) {
    debug_assert!(input.len() == SAMPLE_SIZE);

    // Planes 0-2: unpack bit planes (attackers, defenders, king)
    for plane in 0..3 {
        let base = plane * 16;
        let out_offset = plane * SQS;
        for idx in 0..SQS {
            let byte = idx / 8;
            let bit = idx % 8;
            if (pos.planes[base + byte] >> bit) & 1 == 1 {
                input[out_offset + idx] = 1.0;
            }
        }
    }

    // Plane 3: side to move (all 1s if defenders, stm == 1)
    let stm_val = if pos.stm == 1 { 1.0f32 } else { 0.0f32 };
    let stm_offset = 3 * SQS;
    for i in 0..SQS {
        input[stm_offset + i] = stm_val;
    }

    // Plane 4: throne
    let throne_offset = 4 * SQS;
    input[throne_offset + PRECOMPUTED.throne_sq] = 1.0;

    // Plane 5: corners
    let corners_offset = 5 * SQS;
    for &sq in &PRECOMPUTED.corners_sq {
        input[corners_offset + sq] = 1.0;
    }

    // Plane 6: edge squares
    let edges_offset = 6 * SQS;
    for &sq in &PRECOMPUTED.edges_sq {
        input[edges_offset + sq] = 1.0;
    }

    // Reconstruct piece masks from bit planes for BFS
    let mut king_sq: Option<usize> = None;
    let mut defender_seeds: Vec<usize> = Vec::new();
    let mut group_seeds: Vec<usize> = Vec::new();

    for idx in 0..SQS {
        let byte = idx / 8;
        let bit = idx % 8;
        let is_atk = (pos.planes[byte] >> bit) & 1 == 1;
        let is_def = (pos.planes[16 + byte] >> bit) & 1 == 1;
        let is_king = (pos.planes[32 + byte] >> bit) & 1 == 1;

        if is_king {
            king_sq = Some(idx);
            group_seeds.push(idx);
        } else if is_def {
            defender_seeds.push(idx);
            group_seeds.push(idx);
        }
        let _ = (is_atk, is_def);
    }

    // Plane 7: group BFS — king + all defenders, passable = non-attacker squares
    let group_reach = if !group_seeds.is_empty() {
        bfs(
            |sq| {
                let byte = sq / 8;
                let bit = sq % 8;
                (pos.planes[byte] >> bit) & 1 == 0 // not attacker
            },
            &PRECOMPUTED.vertical_horizontal_neighbors,
            &group_seeds,
        )
    } else {
        [false; SQS]
    };
    let group_offset = 7 * SQS;
    for idx in 0..SQS {
        if group_reach[idx] {
            input[group_offset + idx] = 1.0;
        }
    }

    // Plane 8: king BFS — king only, passable = king's own square or empty (not attacker, not defender)
    let king_reach = if let Some(ksq) = king_sq {
        bfs(
            |sq| {
                let byte = sq / 8;
                let bit = sq % 8;
                let is_atk = (pos.planes[byte] >> bit) & 1 == 1;
                let is_def = (pos.planes[16 + byte] >> bit) & 1 == 1;
                !is_atk && !is_def
            },
            &PRECOMPUTED.vertical_horizontal_neighbors,
            &[ksq],
        )
    } else {
        [false; SQS]
    };
    let king_offset = 8 * SQS;
    // Include king square itself in king BFS plane
    if let Some(ksq) = king_sq {
        input[king_offset + ksq] = 1.0;
    }
    for idx in 0..SQS {
        if king_reach[idx] {
            input[king_offset + idx] = 1.0;
        }
    }

    // Plane 9: repetition x1 — position seen at least once before (rep >= 2)
    let rep1_val = if pos.rep >= 2 { 1.0f32 } else { 0.0 };
    let rep1_offset = 9 * SQS;
    for i in 0..SQS {
        input[rep1_offset + i] = rep1_val;
    }

    // Plane 10: repetition x2 — position seen at least twice before (rep >= 3)
    let rep2_val = if pos.rep >= 3 { 1.0f32 } else { 0.0 };
    let rep2_offset = 10 * SQS;
    for i in 0..SQS {
        input[rep2_offset + i] = rep2_val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::board::position_export::BitPosition;
    use crate::board::types::{Piece, Side};
    use crate::board::utils::get_square_from_algebraic;

    fn sq(alg: &str) -> usize {
        get_square_from_algebraic(alg)
    }

    fn make_input(board: &Board, rep: u8) -> Vec<f32> {
        let bp = BitPosition::from_board(board, rep);
        let mut input = vec![0.0f32; SAMPLE_SIZE];
        fill_input(&mut input, &bp);
        input
    }

    fn pv(input: &[f32], plane: usize, square: usize) -> f32 {
        input[plane * SQS + square]
    }

    #[test]
    fn piece_planes() {
        let mut board = Board::new();
        board.set_piece(sq("a1"), Piece::ATTACKER).unwrap();
        board.set_piece(sq("b2"), Piece::DEFENDER).unwrap();
        board.set_piece(sq("c3"), Piece::KING).unwrap();
        let input = make_input(&board, 1);

        assert_eq!(pv(&input, 0, sq("a1")), 1.0);
        assert_eq!(pv(&input, 0, sq("b2")), 0.0);
        assert_eq!(pv(&input, 1, sq("b2")), 1.0);
        assert_eq!(pv(&input, 1, sq("a1")), 0.0);
        assert_eq!(pv(&input, 2, sq("c3")), 1.0);
        assert_eq!(pv(&input, 2, sq("a1")), 0.0);
    }

    #[test]
    fn stm_plane() {
        let mut board = Board::new();
        board.set_piece(sq("b2"), Piece::KING).unwrap();

        // Attackers to move (default, stm=0) → plane 3 all 0
        let input = make_input(&board, 1);
        assert_eq!(pv(&input, 3, sq("a1")), 0.0);
        assert_eq!(pv(&input, 3, sq("k11")), 0.0);

        // Defenders to move (stm=1) → plane 3 all 1
        board.side_to_move = Side::DEFENDERS;
        let input = make_input(&board, 1);
        assert_eq!(pv(&input, 3, sq("a1")), 1.0);
        assert_eq!(pv(&input, 3, sq("k11")), 1.0);
    }

    #[test]
    fn throne_plane() {
        let board = Board::new();
        let input = make_input(&board, 1);
        assert_eq!(pv(&input, 4, PRECOMPUTED.throne_sq), 1.0);
        assert_eq!(pv(&input, 4, sq("a1")), 0.0);
    }

    #[test]
    fn corners_plane() {
        let board = Board::new();
        let input = make_input(&board, 1);
        for &csq in &PRECOMPUTED.corners_sq {
            assert_eq!(pv(&input, 5, csq), 1.0);
        }
        assert_eq!(pv(&input, 5, sq("b1")), 0.0);
    }

    #[test]
    fn edges_plane() {
        let board = Board::new();
        let input = make_input(&board, 1);
        for &esq in &PRECOMPUTED.edges_sq {
            assert_eq!(pv(&input, 6, esq), 1.0);
        }
        assert_eq!(pv(&input, 6, PRECOMPUTED.throne_sq), 0.0);
    }

    // Validates the bug fix: before the fix, king BFS only lit up the king's own
    // square (same as plane 2). After the fix it expands to all reachable empty squares.
    #[test]
    fn king_bfs_reaches_neighbors() {
        let mut board = Board::new();
        board.set_piece(sq("d4"), Piece::KING).unwrap();
        let input = make_input(&board, 1);

        assert_eq!(pv(&input, 8, sq("d4")), 1.0); // king's own square
        assert_eq!(pv(&input, 8, sq("c4")), 1.0); // left
        assert_eq!(pv(&input, 8, sq("e4")), 1.0); // right
        assert_eq!(pv(&input, 8, sq("d3")), 1.0); // down
        assert_eq!(pv(&input, 8, sq("d5")), 1.0); // up
        assert_eq!(pv(&input, 8, sq("a1")), 1.0); // far corner reachable on empty board
    }

    #[test]
    fn king_bfs_blocked_by_attacker() {
        let mut board = Board::new();
        board.set_piece(sq("d4"), Piece::KING).unwrap();
        board.set_piece(sq("a4"), Piece::ATTACKER).unwrap();
        let input = make_input(&board, 1);

        assert_eq!(pv(&input, 8, sq("b4")), 1.0); // reachable (between king and attacker)
        assert_eq!(pv(&input, 8, sq("a4")), 0.0); // attacker itself not passable
        assert_eq!(pv(&input, 8, sq("k4")), 1.0); // other direction still open
    }

    #[test]
    fn king_bfs_blocked_by_defender() {
        let mut board = Board::new();
        board.set_piece(sq("d4"), Piece::KING).unwrap();
        board.set_piece(sq("a4"), Piece::DEFENDER).unwrap();
        let input = make_input(&board, 1);

        assert_eq!(pv(&input, 8, sq("b4")), 1.0); // reachable
        assert_eq!(pv(&input, 8, sq("a4")), 0.0); // defender blocks king sliding
    }

    #[test]
    fn group_bfs_expands_through_defenders() {
        let mut board = Board::new();
        board.set_piece(sq("a1"), Piece::KING).unwrap();
        board.set_piece(sq("a2"), Piece::DEFENDER).unwrap();
        board.set_piece(sq("b1"), Piece::ATTACKER).unwrap();
        let input = make_input(&board, 1);

        assert_eq!(pv(&input, 7, sq("a1")), 1.0); // king — passable (not attacker)
        assert_eq!(pv(&input, 7, sq("a2")), 1.0); // defender — passable in group BFS
        assert_eq!(pv(&input, 7, sq("a3")), 1.0); // reachable past defender
        assert_eq!(pv(&input, 7, sq("b1")), 0.0); // attacker blocks
    }

    #[test]
    fn group_bfs_attacker_wall_cuts_off_region() {
        let mut board = Board::new();
        board.set_piece(sq("c3"), Piece::KING).unwrap();
        // Vertical wall of attackers cutting off the left side
        for r in &["a1", "b1", "c1", "d1", "e1"] {
            board.set_piece(sq(r), Piece::ATTACKER).unwrap();
        }
        let input = make_input(&board, 1);

        assert_eq!(pv(&input, 7, sq("c3")), 1.0); // king reachable
        // Squares below the attacker wall are cut off from king
        assert_eq!(pv(&input, 7, sq("c1")), 0.0);
    }

    #[test]
    fn rep_planes() {
        let mut board = Board::new();
        board.set_piece(sq("b2"), Piece::KING).unwrap();

        let i1 = make_input(&board, 1);
        assert_eq!(pv(&i1, 9, 0), 0.0);
        assert_eq!(pv(&i1, 10, 0), 0.0);

        let i2 = make_input(&board, 2);
        assert_eq!(pv(&i2, 9, 0), 1.0);
        assert_eq!(pv(&i2, 10, 0), 0.0);

        let i3 = make_input(&board, 3);
        assert_eq!(pv(&i3, 9, 0), 1.0);
        assert_eq!(pv(&i3, 10, 0), 1.0);
    }
}

pub fn build_input_data(positions: &[&BitPosition]) -> Vec<f32> {
    let batch_size = positions.len();
    let mut input_data = vec![0.0f32; batch_size * SAMPLE_SIZE];

    for (i, pos) in positions.iter().enumerate() {
        let start = i * SAMPLE_SIZE;
        let end = start + SAMPLE_SIZE;
        fill_input(&mut input_data[start..end], pos);
    }

    input_data
}
