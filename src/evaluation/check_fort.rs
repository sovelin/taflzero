use crate::board::constants::{BOARD_SIZE, HOLE, SQS};
use crate::board::types::{Piece, Square};
use crate::board::utils::is_edge_square;
use crate::board::{Board, PRECOMPUTED};
use crate::types::OptionalSquare;
use std::collections::{HashSet, VecDeque};


struct Area {
    pub squares: HashSet<Square>,
    pub is_move_possible: bool,
}

struct AreaList {
    areas: Vec<Area>,
    board_map: [Option<usize>; SQS],
}

impl AreaList {
    pub fn new() -> Self {
        Self { areas: vec![], board_map:  [None; SQS] }
    }

    pub fn is_square_in_move_possible_area(&self, sq: Square) -> bool {
        let area_index = self.board_map[sq]
            .expect(&format!("Square {} is not in any area", sq));

        self.areas[area_index].is_move_possible
    }

    pub fn is_square_in_area(&self, sq: Square) -> bool {
        self.board_map[sq].is_some()
    }

    pub fn push_area(&mut self, area: Area) {
        let area_index = self.areas.len();

        for &sq in &area.squares {
            self.board_map[sq] = Some(area_index);
        }

        self.areas.push(area);
    }

    pub fn print(&self) {
        // print in format board:
        // 112110001
        // 111222211

            for row in (0..BOARD_SIZE).rev() {
                for col in 0..BOARD_SIZE {
                    let sq = row * BOARD_SIZE + col;
                    if let Some(area_index) = self.board_map[sq] {
                        print!("{}", area_index);
                    } else {
                        print!(".");
                    }
                }
                println!();
            }


    }
}

fn revert_cleared_defenders(board: &mut Board, cleared_defenders: &Vec<OptionalSquare>) {
    for &sq in cleared_defenders.iter() {
        board.set_piece(sq as Square, Piece::DEFENDER).unwrap();
    }
}

pub fn check_fort(board: &mut Board) -> bool {
    if board.king_sq == HOLE {
        return false;
    }

    if !is_calculate_needed(board) {
        return false;
    }

    if !king_has_moves(board) || !king_contacts_edge(board.king_sq as Square) {
        return false;
    }

    if king_contacts_attackers(&board) {
        return false;
    }

    let mut cleared_defenders: Vec<OptionalSquare> = Vec::new();

    loop {
        let area_list = get_attackers_areas(board);
        area_list.print();
        let broken = try_break_fort(&area_list, board);

        // print board for debug
        println!("After trying to break fort:");
        println!("{:?}", board);

        if broken == HOLE {
            break;
        }

        cleared_defenders.push(broken);

        if king_contacts_attackers(&board) {
            revert_cleared_defenders(board, &cleared_defenders);
            return false;
        }
    }

    revert_cleared_defenders(board, &cleared_defenders);

    true
}

pub fn king_contacts_attackers(board: &Board) -> bool {
    let mut is_contacting = false;

    bfs_ts(
        &[board.king_sq as Square],
        &PRECOMPUTED.vertical_horizontal_neighbors,
        |sq| {
            if board.board[sq] == Piece::ATTACKER {
                is_contacting = true;
            }

            board.board[sq] != Piece::DEFENDER
        },
    );

    is_contacting
}

pub fn get_attackers_areas(board: &Board) -> AreaList {
    let mut available_attackers: HashSet<Square> = board.attackers[..board.attackers_count as usize]
        .iter()
        .cloned()
        .collect();

    let mut areas = AreaList::new();

    while !available_attackers.is_empty()  {
        let next_attacker = *available_attackers.iter().next().unwrap();

        let mut area = Area {
            squares: HashSet::new(),
            is_move_possible: false,
        };

        area.squares.insert(next_attacker);

        bfs_ts(
            [next_attacker].as_slice(),
            &PRECOMPUTED.vertical_horizontal_neighbors,
            |sq| {
                let piece = board.board[sq];

                if piece == Piece::ATTACKER {
                    area.squares.insert(sq);
                    available_attackers.remove(&sq);
                }

                let is_achievable = piece != Piece::DEFENDER && piece != Piece::KING && !PRECOMPUTED.corners_sq.contains(&sq) && sq != PRECOMPUTED.throne_sq;

                if is_achievable {
                    area.squares.insert(sq);

                    if piece == Piece::EMPTY {
                        area.is_move_possible = true;
                    }
                }

                is_achievable
            },
        );

        areas.push_area(area);
        available_attackers.remove(&next_attacker);
    }

    areas
}

fn is_theoretically_possible_to_capture(area_list: &AreaList, a: Option<Square>, b: Option<Square>) -> bool {
    if let (Some(a), Some(b)) = (a, b) {
        if !area_list.is_square_in_area(a) || !area_list.is_square_in_area(b) {
            return false;
        }

        area_list.is_square_in_move_possible_area(a) || area_list.is_square_in_move_possible_area(b)
    } else {
        false
    }
}

fn try_break_fort(area_list: &AreaList, board: &mut Board) -> OptionalSquare {
    for &defender in board.defenders[0..board.defenders_count as usize].iter() {
        // check horizontal capture

        let left = PRECOMPUTED.left_neighbor[defender];
        let right = PRECOMPUTED.right_neighbor[defender];

        if is_theoretically_possible_to_capture(area_list, left, right) {
            board.clear_piece(defender);
            return defender as OptionalSquare;
        }

        // check vertical capture
        let top = PRECOMPUTED.top_neighbor[defender];
        let bottom = PRECOMPUTED.bottom_neighbor[defender];

        if is_theoretically_possible_to_capture(area_list, top, bottom) {
            board.clear_piece(defender);
            return defender as OptionalSquare;
        }
    }

    HOLE
}

fn is_calculate_needed(board: &Board) -> bool {
    if board.last_move_to == HOLE {
        return true;
    };

    let last_move_piece = board.board[board.last_move_to as usize];

    if last_move_piece == Piece::KING {
        return true;
    }

    if last_move_piece == Piece::ATTACKER {
        return false;
    }

    let is_edge = is_edge_square(board.last_move_to as usize);
    let neighbors = &PRECOMPUTED.all_neighbors[board.last_move_to as usize];

    let mut defenders_nearby = 0;
    for &sq in neighbors.iter() {
        if board.board[sq] == Piece::DEFENDER || is_edge_square(sq) {
            defenders_nearby += 1;
        }
    }

    (is_edge && defenders_nearby > 0) || defenders_nearby >= 2
}

fn king_has_moves(board: &Board) -> bool {
    for &neighbor in PRECOMPUTED.vertical_horizontal_neighbors[board.king_sq as usize].iter() {
        if board.board[neighbor] == Piece::EMPTY {
            return true;
        }
    }
    false
}

fn king_contacts_edge(king_sq: Square) -> bool {
    let row = PRECOMPUTED.row[king_sq];
    let col = PRECOMPUTED.col[king_sq];

    row == 0 || row == BOARD_SIZE - 1 || col == 0 || col == BOARD_SIZE - 1
}

fn bfs_ts<F>(
    start_squares: &[Square],
    neighbors: &[Vec<Square>; SQS],
    mut is_achievable: F,
) -> [bool; SQS]
where
    F: FnMut(Square) -> bool,
{
    let mut queue: VecDeque<Square> = VecDeque::new();
    let mut visited = [false; SQS];
    let mut flags = [false; SQS];

    for &sq in start_squares {
        visited[sq] = true;
        flags[sq] = true;
        queue.push_back(sq);
    }

    while let Some(current) = queue.pop_front() {
        for &neighbor in neighbors[current].iter() {
            if visited[neighbor] {
                continue;
            }

            if !is_achievable(neighbor) {
                continue;
            }

            visited[neighbor] = true;
            flags[neighbor] = true;
            queue.push_back(neighbor);
        }
    }

    flags
}

#[cfg(test)]
mod tests {
    use super::check_fort;
    use crate::board::Board;
    use crate::board::types::Piece;
    use crate::board::utils::get_square_from_algebraic;
    use crate::set_board_from_str;

    #[test]
    fn empty_king_not_fort() {
        let mut board = Board::new();
        board
            .set_piece(get_square_from_algebraic("f6"), Piece::KING)
            .unwrap();
        assert!(!check_fort(&mut board));
    }

    #[test]
    fn king_surrounded_by_defenders_in_center_not_fort() {
        let mut board = Board::new();
        board
            .set_piece(get_square_from_algebraic("f6"), Piece::KING)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e6"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e7"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g6"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g7"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("f5"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("f8"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e5"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g5"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e8"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g8"), Piece::DEFENDER)
            .unwrap();
        assert!(!check_fort(&mut board));
    }

    #[test]
    fn king_surrounded_on_edge_impossible_to_break_king_has_moves_fort() {
        let mut board = Board::new();
        board
            .set_piece(get_square_from_algebraic("f1"), Piece::KING)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e1"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g1"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("f3"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e3"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g3"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g8"), Piece::ATTACKER)
            .unwrap();
        assert!(check_fort(&mut board));
    }

    #[test]
    fn king_surrounded_have_access_to_edge_not_on_edge_not_fort() {
        let mut board = Board::new();
        board
            .set_piece(get_square_from_algebraic("f2"), Piece::KING)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e1"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g1"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("f3"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e3"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g3"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g8"), Piece::ATTACKER)
            .unwrap();
        assert!(!check_fort(&mut board));
    }

    #[test]
    fn king_surrounded_on_edge_attacker_inside_not_fort() {
        let mut board = Board::new();
        board
            .set_piece(get_square_from_algebraic("f1"), Piece::KING)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e1"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g1"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("f4"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e3"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g3"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e4"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g4"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("f3"), Piece::ATTACKER)
            .unwrap();
        assert!(!check_fort(&mut board));
    }

    #[test]
    fn king_surrounded_by_attackers_like_fort_not_fort() {
        let mut board = Board::new();
        board
            .set_piece(get_square_from_algebraic("f2"), Piece::KING)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e1"), Piece::ATTACKER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e2"), Piece::ATTACKER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g1"), Piece::ATTACKER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g2"), Piece::ATTACKER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("f3"), Piece::ATTACKER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e3"), Piece::ATTACKER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g3"), Piece::ATTACKER)
            .unwrap();
        assert!(!check_fort(&mut board));
    }

    #[test]
    fn king_surrounded_on_edge_no_moves_not_fort() {
        let mut board = Board::new();
        board
            .set_piece(get_square_from_algebraic("f1"), Piece::KING)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e1"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g1"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("f2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g8"), Piece::ATTACKER)
            .unwrap();
        assert!(!check_fort(&mut board));
    }

    #[test]
    fn king_surrounded_on_edge_possible_to_break_no_moves_not_fort() {
        let mut board = Board::new();
        board
            .set_piece(get_square_from_algebraic("f1"), Piece::KING)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e1"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g1"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("f3"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("b2"), Piece::ATTACKER)
            .unwrap();
        assert!(!check_fort(&mut board));
    }

    #[test]
    fn king_surrounded_on_edge_impossible_to_break_no_moves_complex_case_fort() {
        let mut board = Board::new();
        board
            .set_piece(get_square_from_algebraic("f1"), Piece::KING)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e1"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g1"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("f3"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e3"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("f4"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g3"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("d2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("h2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("i2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("c2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g8"), Piece::ATTACKER)
            .unwrap();
        assert!(check_fort(&mut board));
    }

    #[test]
    fn king_surrounded_on_specific_hard_case_with_holes_is_fort() {
        let mut board = Board::new();
        board
            .set_piece(get_square_from_algebraic("e1"), Piece::KING)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("d1"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("f1"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("d2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("f2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e3"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("c3"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("c4"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("d4"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("b8"), Piece::ATTACKER)
            .unwrap();
        assert!(check_fort(&mut board));

        // check that all defenders are still on board
        assert_eq!(board.defenders_count, 8);
    }

    #[test]
    fn king_surrounded_on_specific_hard_breakable_case_with_holes_is_not_fort() {
        let mut board = Board::new();
        board
            .set_piece(get_square_from_algebraic("e1"), Piece::KING)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("d1"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("f1"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("d2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("f2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e3"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("c3"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("d4"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("b8"), Piece::ATTACKER)
            .unwrap();
        assert!(!check_fort(&mut board));

        // check that all defenders are still on board
        assert_eq!(board.defenders_count, 7);
    }

    #[test]
    fn edge_case_1() {
        let mut board = Board::new();
        board.set_fen("3a3dad1/5dddd1k/9dd/a3a6/a4a1aa2/aa1dd6/a3dd4a/a9a/7d1aa/5a5/3aaaaa3 a").unwrap();

        assert!(!check_fort(&mut board));
    }

    #[test]
    fn edge_case_2() {
        let mut board = Board::new();
        board.set_fen("3a3dad1/5dddd1k/6ddadd/11/a4a1aa2/11/11/11/11/11/11 a").unwrap();

        assert!(!check_fort(&mut board));
    }

    #[test]
    fn edge_case_3() {
        let mut board = Board::new();
        board.set_fen("3a3dad1/5dddd1k/6ddadd/6ddddd/11/11/11/11/11/11/11 a").unwrap();

        assert!(!check_fort(&mut board));
    }


    #[test]
    fn edge_case_4() {
        let mut board = Board::new();
        board.set_fen("7dad1/5dddd1k/6ddadd/6ddddd/11/11/11/11/11/11/11 a").unwrap();

        assert!(check_fort(&mut board));
    }


    #[test]
    fn edge_case_5() {
        let mut board = Board::new();
        board.set_fen("4daadad1/5dddd1k/6ddadd/6ddddd/11/11/11/11/11/11/11 a").unwrap();

        assert!(check_fort(&mut board));
    }

    #[test]
    fn edge_case_6() {
        let mut board = Board::new();
        board.set_fen("4daadad1/5dddd1k/6ddadd/6ddddd/11/11/11/11/11/11/11 a").unwrap();

        assert!(check_fort(&mut board));
    }

    #[test]
    fn edge_case_7() {
        let mut board = Board::new();
        set_board_from_str(
            &mut board,
            ".ADKDAAAAA.
                     AAD.DAAAAAA
                     AAADAAAAAAA
                     AAAAAAAAAAA
                     AAAAAAAAAAA
                     AAAAAAAAAAA
                     AAAAAAAAAAA
                     AAAAAAAAAAA
                     AAAAAAAAAAA
                     AAAAAAAAAAA
                     .AAAAAAAAA."
        );

        println!("{:?}", board);

        assert!(check_fort(&mut board));
    }

    #[test]
    fn edge_case_8() {
        let mut board = Board::new();
        set_board_from_str(
            &mut board,
            ".ADKDAAAAA.
                     AAD.DAAAAAA
                     AAADAAAAAAA
                     AAAAAAAAAAA
                     AAAAAAAAAAA
                     AAAAAAAAAAA
                     AAAAAAAAAAA
                     AAAAAAAAAAA
                     AAAAAAAAAAA
                     AAAAAAAAAAA
                     ..AAAAAAAA."
        );

        println!("{:?}", board);

        assert!(!check_fort(&mut board));
    }

    #[test]
    fn edge_case_9() {
        let mut board = Board::new();
        set_board_from_str(
            &mut board,
            ".ADKDAAAAA.
                     AAD.DAAAAAA
                     AADDAAAAAAA
                     AAAAAAAAAAA
                     AAAAAAAAAAA
                     AAAAAAAAAAA
                     AAAAAAAAAAA
                     AAAAAAAAAAA
                     AAAAAAAAAAA
                     AAAAAAAAAAA
                     ..AAAAAAAA."
        );

        assert!(check_fort(&mut board));
    }
}
