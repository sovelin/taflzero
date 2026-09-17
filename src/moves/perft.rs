//! Move-generation node counter.
//!
//! Counts the leaves of the move tree to a fixed depth. Its only job is to compare
//! against another engine: a mismatch at depth N pins a movegen or capture-rule
//! divergence to a specific position, which no amount of game-playing does as
//! precisely.
//!
//! Terminal positions are deliberately **not** cut off — once the king escapes or
//! falls, counting simply continues. That is what linn's `engine_perft` does, and a
//! node count is only comparable between engines that agree on this.

use crate::board::Board;
use crate::movegen::MoveGen;
use crate::moves::undo::UndoMove;
use crate::mv::Move;

/// Leaves of the move tree at `depth` plies from `board`.
pub fn perft(board: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    // The generator writes into one fixed buffer, so the moves have to be copied out
    // before recursing — the child call would otherwise overwrite them mid-iteration.
    let mut move_gen = MoveGen::new();
    move_gen.generate_moves(board);
    let moves: Vec<Move> = move_gen.moves().to_vec();

    let mut nodes = 0;
    for mv in moves {
        let mut undo = UndoMove::new();
        board.make_move(mv, &mut undo).expect("perft: make_move");
        nodes += perft(board, depth - 1);
        board.unmake_move(&mut undo).expect("perft: unmake_move");
    }

    nodes
}

/// Per-move breakdown at the root, for bisecting a mismatch: compare the two engines'
/// lists, descend into the first move whose count differs, repeat.
pub fn perft_divide(board: &mut Board, depth: u32) -> Vec<(Move, u64)> {
    if depth == 0 {
        return Vec::new();
    }

    let mut move_gen = MoveGen::new();
    move_gen.generate_moves(board);
    let moves: Vec<Move> = move_gen.moves().to_vec();

    let mut result = Vec::with_capacity(moves.len());
    for mv in moves {
        let mut undo = UndoMove::new();
        board.make_move(mv, &mut undo).expect("perft: make_move");
        let nodes = perft(board, depth - 1);
        board.unmake_move(&mut undo).expect("perft: unmake_move");
        result.push((mv, nodes));
    }

    result
}
