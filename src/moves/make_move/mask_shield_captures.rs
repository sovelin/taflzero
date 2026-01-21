use crate::board::{get_side_by_piece, Board, PRECOMPUTED};
use crate::board::constants::BOARD_SIZE;
use crate::board::types::{Piece, Side, Square};
use crate::board::utils::{get_col, get_row};
use crate::moves::undo::{CapturedPiece, UndoMove};
use crate::types::OptionalSquare;

#[derive(Copy, Clone)]
pub enum ShieldSide { Top, Bottom, Left, Right }

#[inline]
fn shield_sides(to_sq: Square) -> Vec<ShieldSide> {
    use ShieldSide::*;
    let mut v = Vec::<ShieldSide>::new();
    let r = PRECOMPUTED.row[to_sq];
    let c = PRECOMPUTED.col[to_sq];

    if r <= 1 { v.push(Bottom); }
    if r >= BOARD_SIZE - 2 { v.push(Top); }
    if c <= 1 { v.push(Left); }
    if c >= BOARD_SIZE - 2 { v.push(Right); }
    v
}

type ShieldItertor = (
    fn() -> Square,
    fn(Square) -> bool,
    fn(Square) -> Option<Square>,
    fn(Square) -> Option<Square>,
    fn(Square) -> bool,
);

// ===== ядро: поиск взятий вдоль конкретной стороны щита =====
fn captures_on_side(board: &mut Board, side: Side, which: ShieldSide, undo: &mut UndoMove) {
    // Для каждой стороны задаём функции “направлений” как замыкания.
    // next — как мы движемся вдоль края; roof — клетка “над” проверяемой фигурой.
    let (start, is_last, next, roof, is_always_friend): ShieldItertor = match which {
        ShieldSide::Top => (
            || PRECOMPUTED.top_left_sq,
            |sq: Square| sq == PRECOMPUTED.top_right_sq,
            |sq: Square| PRECOMPUTED.right_neighbor[sq],
            |sq: Square| PRECOMPUTED.bottom_neighbor[sq],
            |sq: Square| sq == PRECOMPUTED.top_left_sq || sq == PRECOMPUTED.top_right_sq,
        ),
        ShieldSide::Bottom => (
            || PRECOMPUTED.bottom_left_sq,
            |sq: Square| sq == PRECOMPUTED.bottom_right_sq,
            |sq: Square| PRECOMPUTED.right_neighbor[sq],
            |sq: Square| PRECOMPUTED.top_neighbor[sq],
            |sq: Square| sq == PRECOMPUTED.bottom_left_sq || sq == PRECOMPUTED.bottom_right_sq,
        ),
        ShieldSide::Left => (
            || PRECOMPUTED.top_left_sq,
            |sq: Square| sq == PRECOMPUTED.bottom_left_sq,
            |sq: Square| PRECOMPUTED.bottom_neighbor[sq],
            |sq: Square| PRECOMPUTED.right_neighbor[sq],
            |sq: Square| sq == PRECOMPUTED.top_left_sq || sq == PRECOMPUTED.bottom_left_sq,
        ),
        ShieldSide::Right => (
            || PRECOMPUTED.top_right_sq,
            |sq: Square| sq == PRECOMPUTED.bottom_right_sq,
            |sq: Square| PRECOMPUTED.bottom_neighbor[sq],
            |sq: Square| PRECOMPUTED.left_neighbor[sq],
            |sq: Square| sq == PRECOMPUTED.top_right_sq || sq == PRECOMPUTED.bottom_right_sq,
        ),
    };

    #[inline]
    fn is_friend(board: &Board, side: Side, sq: Square) -> bool {
        match get_side_by_piece(board.board[sq]) {
            Ok(s) => s == side,
            Err(_) => false,
        }
    }

    let mut res: Vec<Square> = Vec::new();
    let mut seq: Vec<Square> = Vec::new();
    let mut seq_started = true;

    // начинаем “до” первой клетки: как в TS, первая итерация — это next(start)
    let mut it = Some(start());

    fn add_to_undo_and_remove(sq: Square, board: &mut Board, undo: &mut UndoMove) {
        if board.board[sq] == Piece::KING {
            return;
        }

        let piece = board.board[sq];
        undo.add_captured_piece(CapturedPiece { square: sq, piece });
        board.clear_piece(sq);

    }

    let mut start_sq: usize = 0;

    while let Some(next_sq) = it.and_then(next) {
        // Дошли до последней клетки ряда: финализируем текущую серию
        if is_last(next_sq) {
            if seq.len() > 1 {
                res.extend_from_slice(&seq);
                seq.clear();
            }
            // цикл продолжится, но дальше next(last) должен вернуть None
        } else if board.board[next_sq] == Piece::EMPTY && !is_always_friend(next_sq) {
            // пустая — обнуляем серию
            seq.clear();
            seq_started = false;
        } else if is_friend(board, side, next_sq) || is_always_friend(next_sq) {
            // своя фигура — закрываем серию, если >=2
            if seq.len() > 1 {
                if board.last_move_to == start_sq as OptionalSquare || board.last_move_to == next_sq as OptionalSquare {
                res.extend_from_slice(&seq);
                }
                seq.clear();
                println!("Start: capture sequence at {:?}", start_sq);
                println!("End: capture sequence at {:?}", next_sq);
            }
            seq_started = true;
            start_sq = next_sq;
            println!("start_sq: {:?}", start_sq);
        } else {
            // чужая фигура: проверяем “крышу”
            if !seq_started {
                // серия ещё не началась (перед этим была дыра) — пропускаем
            } else if let Some(roof_sq) = roof(next_sq) {
                if is_friend(board, side, roof_sq) {
                    seq.push(next_sq);
                } else {
                    // крыши нет/враги — обнуляем серию
                    seq.clear();
                    seq_started = false;
                }
            } else {
                // нет крыши — серия ломается
                seq.clear();
                seq_started = false;
            }
        }

        it = Some(next_sq);
    }

    for sq in &res {
        add_to_undo_and_remove(*sq, board, undo);
    }
}

fn is_edged_sq(sq: Square) -> bool {
    let row = get_row(sq);
    let col = get_col(sq);

    row == 0 || row == BOARD_SIZE - 1 || col == 0 || col == BOARD_SIZE - 1
}

pub fn make_shield_wall_captures(board: &mut Board, to_sq: Square, undo: &mut UndoMove) {
    if !is_edged_sq(to_sq) {
        return;
    }

    let sides = shield_sides(to_sq);
    if sides.is_empty() { return; }

    for s in sides {
        captures_on_side(board, board.side_to_move, s, undo);
    }
}
