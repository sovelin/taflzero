use rand::Rng;
use crate::board::Board;
use crate::board::types::Side;
use crate::board::utils::{get_col, get_row};
use crate::evaluation::{sided_evaluation, MATE_SCORE};
use crate::evaluation::terminal::check_terminal;
use crate::is_mate_score;
use crate::moves::mv::Move;
use crate::types::Square;
use super::search_data::SearchData;
use super::transposition::{TTFlag, TranspositionTable};

fn is_quiet(search_data: &mut SearchData, height: u32, mv: Move, board: &Board, is_capture: bool) -> bool {
    let king_row = get_row(board.king_sq as Square);
    let king_col = get_col(board.king_sq as Square);
    let move_row = get_row(mv.to());
    let move_col = get_col(mv.to());
    let is_king_move = mv.from() == board.king_sq as Square;
    let is_adjacent_to_king = (king_row as i32 - move_row as i32).abs() <= 1 && (king_col as i32 - move_col as i32).abs() <= 1;
    let is_influencing_to_king = king_col == move_col || king_row == move_row;

    search_data.killers.get(height as usize)[0] != mv
        && search_data.killers.get(height as usize)[1] != mv
        && !is_capture
        && !is_king_move
        && !is_adjacent_to_king
        && !is_influencing_to_king }

pub fn search(
    board: &mut Board,
    depth: u32,
    mut alpha: i32,
    mut beta: i32,
    height: u32,
    search_data: &mut SearchData,
    tt: &mut TranspositionTable,
) -> i32 {
    search_data.nodes_searched += 1;
    if let Some(result) = check_terminal(board) {
        let score = if result == Side::DEFENDERS {
            MATE_SCORE - height as i32
        } else {
            -MATE_SCORE + height as i32
        };

        return sided_evaluation(score, board.side_to_move);
    }

    if depth == 0 {
        return board.get_eval();
    }

    let is_pv_node = alpha < beta - 1;
    let tt_entry = tt.probe(board.zobrist);

    if !is_pv_node && tt_entry.is_valid(board.zobrist) && tt_entry.depth() as u32 >= depth {
        let tt_score = tt_entry.score(height);

        match tt_entry.flag() {
            TTFlag::Exact => {
                return tt_score;
            }
            TTFlag::LowerBound => {
                alpha = alpha.max(tt_score);
            }
            TTFlag::UpperBound => {
                beta = beta.min(tt_score);
            }
        }

        if alpha >= beta {
            return tt_score;
        }
    }


    let mut tt_type = TTFlag::UpperBound;
    let mut tt_move = Move::default();


    if search_data.time_exceeded() {
        return 0;
    }

    search_data.move_gens[height as usize].generate_moves(board);

    search_data.move_gens[height as usize].order_moves(
        board,
        tt_entry.best_move(),
        &search_data.killers,
        &search_data.history,
        height as usize,
    );

    let mut moves_count = 0;

    if height == 0 && search_data.temperature > 0 {
        for i in 0..search_data.move_gens[height as usize].count() {
            search_data.temperatures[height as usize][i] = search_data.random_generator.gen_range(0..search_data.temperature as i32);
        }
    }

    while let Some(mv) = search_data.move_gens[height as usize].pick_move() {
        let bonus = if is_mate_score(alpha) {0} else { search_data.temperatures[height as usize][moves_count as usize] };
        moves_count += 1;

        // let is_quiet = search_data.killers.get(height as usize)[0] != mv
        //     && search_data.killers.get(height as usize)[1] != mv;


        let piece_count = board.attackers_count + board.defenders_count;
        board.make_move(mv, &mut search_data.undos[height as usize]).unwrap();
        let piece_count_after = board.attackers_count + board.defenders_count;
        let is_capture = piece_count_after < piece_count;

        let mut score: i32;

        let is_lmr = !is_pv_node && depth >= 2 && moves_count > 1 && is_quiet(
            search_data,
            height,
            mv,
            board,
            is_capture,
        );

        if is_pv_node && moves_count == 1 {
            score = -search(
                board,
                depth - 1,
                -beta + bonus,
                -alpha + bonus,
                height + 1,
                search_data,
                tt,
            );
        } else if is_lmr {
            score = -search(
                board,
                depth - 1 - 1,
                -alpha - 1 + bonus,
                -alpha + bonus,
                height + 1,
                search_data,
                tt,
            );

            if score > alpha {
                score = -search(
                    board,
                    depth - 1,
                    -beta + bonus,
                    -alpha + bonus,
                    height + 1,
                    search_data,
                    tt,
                );
            }
        } else {
            score = -search(
                board,
                depth - 1,
                -alpha - 1 + bonus,
                -alpha + bonus,
                height + 1,
                search_data,
                tt,
            );

            if score > alpha && score < beta {
                score = -search(
                    board,
                    depth - 1,
                    -beta + bonus,
                    -alpha + bonus,
                    height + 1,
                    search_data,
                    tt,
                );
            }
        }

        board.unmake_move(&mut search_data.undos[height as usize]).unwrap();

        if  search_data.time_exceeded() {
            return 0;
        }

        if score > alpha {
            alpha = score;

            if height == 0 {
                search_data.best_move = Some(mv);
            }

            tt_type = TTFlag::Exact;
            tt_move = mv;
            search_data.history.update(board.side_to_move, mv, depth as i32);

            if score >= beta {
                tt_type = TTFlag::LowerBound;
                search_data.killers.save(height as usize, mv);
                break;
            }
        }
    }

    tt.store(
        board.zobrist,
        depth as u8,
        alpha,
        tt_type,
        tt_move,
        height,
        search_data.tt_age,
    );

    alpha
}

pub fn qsearch(
    board: &mut Board,
    mut alpha: i32,
    mut beta: i32,
    height: u32,
    search_data: &mut SearchData,
    tt: &mut TranspositionTable,
) -> i32 {
    search_data.nodes_searched += 1;
    if let Some(result) = check_terminal(board) {
        let score = if result == Side::DEFENDERS {
            MATE_SCORE - height as i32
        } else {
            -MATE_SCORE + height as i32
        };

        return sided_evaluation(score, board.side_to_move);
    }

    let stand_pat = board.get_eval();

    if stand_pat >= beta {
        return beta;
    }

    if stand_pat > alpha {
        alpha = stand_pat;
    }

    if search_data.time_exceeded() {
        return 0;
    }

    search_data.capture_gens[height as usize].generate_captures(board);

    for i in 0..search_data.capture_gens[height as usize].count {
        let mv = search_data.capture_gens[height as usize].captures[i];

        board.make_move(mv, &mut search_data.undos[height as usize]).unwrap();

        let score = -qsearch(
            board,
            -beta,
            -alpha,
            height + 1,
            search_data,
            tt,
        );

        board.unmake_move(&mut search_data.undos[height as usize]).unwrap();

        if  search_data.time_exceeded() {
            return 0;
        }

        if score > alpha {
            alpha = score;

            if score >= beta {
                break;
            }
        }
    }

    alpha
}