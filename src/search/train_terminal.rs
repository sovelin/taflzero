use crate::{Board, PRECOMPUTED};
use crate::board::constants::{BOARD_SIZE, HOLE, SQS};
use crate::rules::RulesEnum;
use crate::types::{Piece, Side, Square};
use crate::utils::bfs::bfs;

pub fn check_copenhagen_train_terminal(board: &Board) -> Option<Side> {
    if board.rules != RulesEnum::Copenhagen11x11 {
        return None;
    }

    let mut start_sqs = vec![];

    if board.king_sq != HOLE {
        start_sqs.push(board.king_sq as Square);
    }

    let defenders_count = board.defenders_count as usize;

    for i in 0..board.defenders_count {
        let sq = board.defenders[i as usize];
        start_sqs.push(sq);
    }

    let defenders_bfs = bfs(
        |sq| board.board[sq] != Piece::ATTACKER,
        &PRECOMPUTED.vertical_horizontal_neighbors,
        &start_sqs
    );

    if defenders_bfs[PRECOMPUTED.corners_sq[0] as usize]
        || defenders_bfs[PRECOMPUTED.corners_sq[1] as usize]
        || defenders_bfs[PRECOMPUTED.corners_sq[2] as usize]
        || defenders_bfs[PRECOMPUTED.corners_sq[3] as usize]
    {
        return None;
    }

    // BFS from each corner through non-attacker squares, merged into one reachability map.
    // Squares NOT in this map are isolated from all corners by attackers — potential fort zones.
    let mut corner_reachable = [false; SQS];
    for &corner in &PRECOMPUTED.corners_sq {
        let corner_bfs = bfs(
            |sq| board.board[sq] != Piece::ATTACKER,
            &PRECOMPUTED.vertical_horizontal_neighbors,
            &[corner],
        );
        for i in 0..SQS {
            corner_reachable[i] |= corner_bfs[i];
        }
    }

    // Find the longest run of empty squares on an edge that are cut off from all corners.
    let max_free_run = |edge: &[Square]| -> usize {
        let mut max_run = 0usize;
        let mut run = 0usize;
        for &sq in edge {
            if board.board[sq] == Piece::EMPTY && !corner_reachable[sq] {
                run += 1;
                max_run = max_run.max(run);
            } else {
                run = 0;
            }
        }
        max_run
    };

    let n = BOARD_SIZE;
    let bottom: Vec<Square> = (0..n).collect();
    let top: Vec<Square>    = (0..n).map(|c| (n - 1) * n + c).collect();
    let left: Vec<Square>   = (0..n).map(|r| r * n).collect();
    let right: Vec<Square>  = (0..n).map(|r| r * n + (n - 1)).collect();

    let max_free = [&bottom, &top, &left, &right]
        .iter()
        .map(|e| max_free_run(e))
        .max()
        .unwrap_or(0);

    // Gap of 4 → fort with a single closing row of 4 defenders; gap of 3 → needs 5 defenders to
    // enclose on two rows. In both cases king is also required (implicitly present if we got here).
    if max_free >= 4 && defenders_count >= 4 {
        return None;
    }
    if max_free >= 3 && defenders_count >= 5 {
        return None;
    }

    Some(Side::ATTACKERS)
}

// tests

#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::rules::RulesEnum;
    use crate::search::train_terminal::check_copenhagen_train_terminal;
    use crate::set_board_from_str;
    use crate::types::Side;

    #[test]
    fn cant_build_any_fort_due_to_piece_count() {
        let mut board = Board::new();
        set_board_from_str(
            &mut board,
            "...A....A..
                    ..A......A.
                    AA........A
                    ...........
                    A..........
                    .........K.
                    .......D...
                    .....A....A
                    A........A.
                    .A.......A.
                    ..A...D.A..",
        );

        let res = check_copenhagen_train_terminal(&board);

        assert_eq!(res, Some(Side::ATTACKERS));
    }

    #[test]
    fn corner_is_open_game_continues() {
        let mut board = Board::new();
        set_board_from_str(
            &mut board,
            "...A....A..
                    ..A......A.
                    AA........A
                    ...........
                    A..........
                    .........K.
                    .......D...
                    .....A....A
                    A........A.
                    .........A.
                    ..A...D.A..",
        );

        let res = check_copenhagen_train_terminal(&board);

        assert_eq!(res, None);
    }

    #[test]
    fn can_build_any_fort() {
        let mut board = Board::new();
        set_board_from_str(
            &mut board,
            "...A....A..
                    ..A......A.
                    AA........A
                    ....D......
                    A..D.......
                    .........K.
                    .......D...
                    .....A....A
                    A........A.
                    .A.......A.
                    ..A...D.A..",
        );

        let res = check_copenhagen_train_terminal(&board);

        assert_eq!(res, None);
    }

    #[test]
    fn cant_build_low_fort_due_to_pieces_count() {
        let mut board = Board::new();
        set_board_from_str(
            &mut board,
            "...A...AA..
                    ..A......A.
                    AA........A
                    A...D.....A
                    ...D.......
                    A........K.
                    .A.....D...
                    ..A..A....A
                    ...A.....A.
                    ....A....A.
                    .....ADAA..",
        );

        let res = check_copenhagen_train_terminal(&board);

        assert_eq!(res, Some(Side::ATTACKERS));
    }

    #[test]
    fn can_build_compact_fort() {
        let mut board = Board::new();
        set_board_from_str(
            &mut board,
            "...A...AA..
                    ..A......A.
                    AA........A
                    A...D.....A
                    ...D.......
                    A.D......K.
                    .A.....D...
                    ..A..A....A
                    ...A.....A.
                    ....A....A.
                    .....ADAA..",
        );

        let res = check_copenhagen_train_terminal(&board);

        assert_eq!(res, None);
    }

    #[test]
    fn game_continues_when_defenders_near_corner() {
        let mut board = Board::new();
        set_board_from_str(
            &mut board,
            "...A...AA..
                    ..A......A.
                    AA........A
                    A.........A
                    ...........
                    A.D......K.
                    .A.........
                    ..A..A....A
                    ...A.....A.
                    .D..A....A.
                    .....ADAA..",
        );

        let res = check_copenhagen_train_terminal(&board);

        assert_eq!(res, None);
    }

    #[test]
    fn cant_build_any_fort() {
        let mut board = Board::new();
        set_board_from_str(
            &mut board,
            "....A.A....
                    ...A...A...
                    ..A.....A..
                    .A.......A.
                    A....D....A
                    ....D.D..K.
                    A..D...D...
                    .A........A
                    ..A......A.
                    ...A....A..
                    ....A..A...",
        );

        let res = check_copenhagen_train_terminal(&board);

        assert_eq!(res, Some(Side::ATTACKERS));
    }

    #[test]
    fn game_continues_when_king_near_corner() {
        let mut board = Board::new();
        set_board_from_str(
            &mut board,
            "...A...AA..
                    ..A......A.
                    AA........A
                    A.........A
                    ...........
                    A.D........
                    .A.........
                    ..A..A....A
                    ...A.....A.
                    .K..A....A.
                    .....ADAA..",
        );

        let res = check_copenhagen_train_terminal(&board);

        assert_eq!(res, None);
    }

    #[test]
    fn not_test_other_variants() {
        let mut board = Board::new();
        board.set_rules(RulesEnum::Historical11x11);
        set_board_from_str(
            &mut board,
            "...A....A..
                    ..A......A.
                    AA........A
                    ...........
                    A..........
                    .........K.
                    .......D...
                    .....A....A
                    A........A.
                    .A.......A.
                    ..A...D.A..",
        );

        let res = check_copenhagen_train_terminal(&board);

        assert_eq!(res, None);
    }
}