//! Per-position features beyond the 11 base planes (docs/plan-T7-planes.md).
//!
//! Shared by the network input of nets with extra planes (`NnInput`), the self-play
//! side file written next to every `workerN.bin`, and the replay tool
//! (`taflzero --features`, feature_dump.rs). The 144-byte record is the side file's
//! unit; `ExtraFeatures` is the same content held in memory for one position.

use crate::board::Board;
use crate::board::types::{Piece, Side, Square};
use crate::mcts::export::FLAG_RESULT_UNKNOWN;
use crate::movegen::MoveGen;
use crate::mv::Move;
use crate::terminal::{TerminalType, check_terminal, get_terminal};
use crate::undo::UndoMove;

pub const FEAT_MAGIC: &[u8; 8] = b"TZFEAT01";
pub const FEAT_HEADER_BYTES: usize = 16;
pub const FEAT_RECORD_BYTES: usize = 144;

/// `hist` entry for a ply that does not exist (the start of the game).
pub const NO_SQUARE: u8 = 255;

/// Record `flags`: the move played from this position moves the king.
pub const FEAT_KING_MOVE: u8 = 1 << 0;
/// The move played from this position captures at least one piece.
pub const FEAT_CAPTURE: u8 = 1 << 1;
/// This sample's move was the game's last ply.
pub const FEAT_LAST_PLY: u8 = 1 << 2;
/// The game ended by threefold repetition.
pub const FEAT_THREEFOLD: u8 = 1 << 3;

/// The sample was searched after the game was decided (soft resignation): cheap search,
/// trained at a reduced weight.
pub const FEAT2_REDUCED: u8 = 1 << 0;
/// KL above this many nats is stored as the maximum.
pub const SURPRISE_MAX: f32 = 4.0;
/// Quantization of the surprise byte: 255 / SURPRISE_MAX.
pub const SURPRISE_SCALE: f32 = 63.75;

/// The surprise byte for a KL divergence in nats.
pub fn surprise_byte(kl: f32) -> u8 {
    if !kl.is_finite() || kl <= 0.0 {
        return 0;
    }
    (kl.min(SURPRISE_MAX) * SURPRISE_SCALE).round() as u8
}

/// Fill the self-play-only bytes (surprise, FLAGS2) of `records`, one record per sample.
pub fn set_training_bytes(records: &mut [u8], surprise: &[u8], flags2: &[u8]) {
    let n = records.len() / FEAT_RECORD_BYTES;
    assert!(
        surprise.len() == n && flags2.len() == n,
        "one value per record"
    );
    for i in 0..n {
        let r = &mut records[i * FEAT_RECORD_BYTES..(i + 1) * FEAT_RECORD_BYTES];
        r[offset::SURPRISE] = surprise[i];
        r[offset::FLAGS2] = flags2[i];
    }
}

/// Record offsets (see the table in docs/probe-input-planes.md).
pub mod offset {
    pub const HIST: usize = 0;
    pub const REP3: usize = 8;
    pub const REP2: usize = 24;
    pub const KING1: usize = 40;
    pub const KING2: usize = 56;
    pub const PLY: usize = 72;
    pub const PLIES_TO_END: usize = 74;
    pub const FLAGS: usize = 76;
    pub const TERMINAL: usize = 77;
    /// Policy surprise of the sample's search, KL(target || net prior) in nats, capped
    /// at `SURPRISE_MAX` and quantized as `round(kl * SURPRISE_SCALE)`; 0 for cheap
    /// searches. Written by self-play only: the replay tool cannot know it and writes 0.
    pub const SURPRISE: usize = 78;
    /// FEAT2_* bits, self-play only like SURPRISE.
    pub const FLAGS2: usize = 79;
    pub const CAP_NOW: usize = 80;
    pub const CAP_NEXT: usize = 96;
    pub const ATK_MOB: usize = 112;
    pub const DEF_MOB: usize = 128;
}

/// Square set packed like `BitPosition.planes`: square `i` is bit `i % 8` of byte `i / 8`.
pub type SquareMask = [u8; 16];

#[inline]
pub fn set_bit(mask: &mut SquareMask, sq: Square) {
    mask[sq / 8] |= 1 << (sq % 8);
}

#[inline]
pub fn has_bit(mask: &SquareMask, sq: Square) -> bool {
    (mask[sq / 8] >> (sq % 8)) & 1 == 1
}

pub fn popcount(mask: &SquareMask) -> u32 {
    mask.iter().map(|b| b.count_ones()).sum()
}

/// The history-free part of a record: everything that follows from the board and its
/// repetition table alone.
#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub struct PositionFeatures {
    pub rep3: SquareMask,
    pub rep2: SquareMask,
    pub king1: SquareMask,
    pub king2: SquareMask,
    pub cap_now: SquareMask,
    pub cap_next: SquareMask,
    pub atk_mob: SquareMask,
    pub def_mob: SquareMask,
}

/// Plays every legal move of one side on `board` and takes it back, collecting
/// destinations, captured squares and (for the side to move) how often the resulting
/// position has been seen. The board is left exactly as it was.
fn scan_side(
    board: &mut Board,
    mg: &mut MoveGen,
    moves: &mut Vec<Move>,
    mobility: &mut SquareMask,
    captures: &mut SquareMask,
    mut repeats: Option<(&mut SquareMask, &mut SquareMask)>,
) {
    mg.generate_moves(board);
    moves.clear();
    moves.extend_from_slice(mg.moves());
    let mut undo = UndoMove::new();
    for &mv in moves.iter() {
        set_bit(mobility, mv.to());
        board
            .make_move(mv, &mut undo)
            .expect("generated move failed to apply");
        for c in undo.captured_pieces() {
            set_bit(captures, c.square);
        }
        // The board's own table, which is what the threefold terminal reads: a count
        // of 3 after the move means the move ends the game as an attacker win.
        if let Some((rep3, rep2)) = repeats.as_mut() {
            match board.rep_table.get(&board.zobrist).copied().unwrap_or(0) {
                n if n >= 3 => set_bit(rep3, mv.to()),
                2 => set_bit(rep2, mv.to()),
                _ => {}
            }
        }
        board
            .unmake_move(&mut undo)
            .expect("unmake of a generated move failed");
    }
}

/// Features of the position on `board` (side to move as on the board). `board` is
/// restored before returning.
pub fn position_features(board: &mut Board, mg: &mut MoveGen) -> PositionFeatures {
    let mut f = PositionFeatures::default();
    let mut moves: Vec<Move> = Vec::with_capacity(256);
    let stm = board.side_to_move;

    let (stm_mob, opp_mob) = match stm {
        Side::ATTACKERS => (&mut f.atk_mob, &mut f.def_mob),
        Side::DEFENDERS => (&mut f.def_mob, &mut f.atk_mob),
    };
    scan_side(
        board,
        mg,
        &mut moves,
        stm_mob,
        &mut f.cap_now,
        Some((&mut f.rep3, &mut f.rep2)),
    );
    let stm_moves = moves.clone();

    // A null move: the opponent's threats in the position as it stands.
    board.flip_side();
    scan_side(board, mg, &mut moves, opp_mob, &mut f.cap_next, None);
    board.flip_side();

    if board.king_sq >= 0 {
        let king = board.king_sq as Square;
        let defender_moves = if stm == Side::DEFENDERS {
            &stm_moves
        } else {
            &moves
        };
        for mv in defender_moves.iter().filter(|mv| mv.from() == king) {
            set_bit(&mut f.king1, mv.to());
        }

        // Second king move over the board as it stands: the king alone is moved, no
        // captures are made (the usual approximation for a two-move reach plane).
        let firsts: Vec<Square> = (0..board.sqs())
            .filter(|&sq| has_bit(&f.king1, sq))
            .collect();
        for s in firsts {
            board.clear_piece(king);
            board.set_piece(s, Piece::KING).expect("place king");
            mg.generate_moves_from(board, s);
            for mv in mg.moves() {
                let to = mv.to();
                if to != king && !has_bit(&f.king1, to) {
                    set_bit(&mut f.king2, to);
                }
            }
            board.clear_piece(s);
            board.set_piece(king, Piece::KING).expect("restore king");
        }
    }
    f
}

/// What the last move of a decided game was, as far as the result tells: every legal
/// move from the last sample that ends the game with the recorded winner.
struct LastMove {
    king: bool,
    capture: bool,
    threefold: bool,
    terminal: u8,
}

/// Record byte 77 on a game's last sample: how the game ended. The names are the
/// gamelog's terminal strings (`TerminalType::name` and the datagen's own no-moves
/// and limit endings); dataset.py's FEAT_TERMINAL_NAMES mirrors this table.
pub const TERMINAL_CODES: [(u8, &str); 9] = [
    (1, "atk_capture"),
    (2, "atk_surrounded"),
    (3, "atk_no_moves"),
    (4, "atk_threefold"),
    (5, "def_corner"),
    (6, "def_edge"),
    (7, "def_fort"),
    (8, "def_no_moves"),
    (9, "max_plies"),
];
/// The game was cut off by the ply or no-capture limit.
pub const TERMINAL_LIMIT: u8 = 9;
/// Decided, but the replay could not tell how (candidate moves disagree).
pub const TERMINAL_UNKNOWN: u8 = 255;

/// Code of a gamelog terminal name; both limit endings map to `TERMINAL_LIMIT`.
pub fn terminal_code(name: &str) -> u8 {
    if name.starts_with("undecided") || name == "max_plies" {
        return TERMINAL_LIMIT;
    }
    TERMINAL_CODES
        .iter()
        .find(|(_, n)| *n == name)
        .map(|(c, _)| *c)
        .unwrap_or(TERMINAL_UNKNOWN)
}

/// Gamelog name of a terminal code, `None` for 0 / unknown.
pub fn terminal_name(code: u8) -> Option<&'static str> {
    TERMINAL_CODES
        .iter()
        .find(|(c, _)| *c == code)
        .map(|(_, n)| *n)
}

fn resolve_last_move(board: &mut Board, last_flags: u8, last_value: i8) -> Vec<LastMove> {
    if last_flags & FLAG_RESULT_UNKNOWN != 0 || last_value == 0 {
        return Vec::new();
    }
    let stm = board.side_to_move;
    let winner = if last_value > 0 {
        stm
    } else {
        Side::opposite(stm)
    };
    let mut mg = MoveGen::new();
    mg.generate_moves(board);
    let moves: Vec<Move> = mg.moves().to_vec();
    let mut undo = UndoMove::new();
    let mut out = Vec::new();
    for mv in moves {
        let king = board.board[mv.from()] == Piece::KING;
        board.make_move(mv, &mut undo).expect("generated move");
        let capture = undo.captured_pieces_count > 0;
        let hit = match get_terminal(board) {
            Some(t) => (check_terminal(board) == Some(winner)).then(|| {
                (
                    t == TerminalType::ThreefoldRepetition,
                    terminal_code(t.name()),
                )
            }),
            // Self-play also ends a game when the side to move has no move at all
            // (the mover's side wins; named after the winner as in the gamelog).
            None => {
                mg.generate_moves(board);
                (mg.count == 0 && winner == stm).then(|| {
                    let name = match stm {
                        Side::ATTACKERS => "atk_no_moves",
                        Side::DEFENDERS => "def_no_moves",
                    };
                    (false, terminal_code(name))
                })
            }
        };
        board.unmake_move(&mut undo).expect("unmake");
        if let Some((threefold, terminal)) = hit {
            out.push(LastMove {
                king,
                capture,
                threefold,
                terminal,
            });
        }
    }
    out
}

pub fn write_record(
    out: &mut Vec<u8>,
    f: &PositionFeatures,
    hist: [(u8, u8); 4],
    ply: usize,
    plies_to_end: usize,
    flags: u8,
    terminal: u8,
) {
    let mut r = [0u8; FEAT_RECORD_BYTES];
    r[offset::TERMINAL] = terminal;
    for (k, (from, to)) in hist.iter().enumerate() {
        r[offset::HIST + 2 * k] = *from;
        r[offset::HIST + 2 * k + 1] = *to;
    }
    r[offset::REP3..offset::REP3 + 16].copy_from_slice(&f.rep3);
    r[offset::REP2..offset::REP2 + 16].copy_from_slice(&f.rep2);
    r[offset::KING1..offset::KING1 + 16].copy_from_slice(&f.king1);
    r[offset::KING2..offset::KING2 + 16].copy_from_slice(&f.king2);
    r[offset::PLY..offset::PLY + 2]
        .copy_from_slice(&(ply.min(u16::MAX as usize) as u16).to_le_bytes());
    r[offset::PLIES_TO_END..offset::PLIES_TO_END + 2]
        .copy_from_slice(&(plies_to_end.min(u16::MAX as usize) as u16).to_le_bytes());
    r[offset::FLAGS] = flags;
    r[offset::CAP_NOW..offset::CAP_NOW + 16].copy_from_slice(&f.cap_now);
    r[offset::CAP_NEXT..offset::CAP_NEXT + 16].copy_from_slice(&f.cap_next);
    r[offset::ATK_MOB..offset::ATK_MOB + 16].copy_from_slice(&f.atk_mob);
    r[offset::DEF_MOB..offset::DEF_MOB + 16].copy_from_slice(&f.def_mob);
    out.extend_from_slice(&r);
}

/// How a game's last move was read off its result (see `finish_game`).
#[derive(Default, Debug, Clone, PartialEq)]
pub struct GameEnd {
    pub threefold: bool,
    /// Decided game, but no legal move from the last sample reaches the result.
    pub unresolved: bool,
    /// Several moves reach it and disagree on the king / capture / threefold bits.
    pub ambiguous: bool,
    /// Final record flags per sample.
    pub flags: Vec<u8>,
    /// Byte 77 of the last record.
    pub terminal: u8,
}

/// Writes a finished game's records. `features[i]` and `move_flags[i]` (king move,
/// capture of the move played from sample i) are per sample, `played` the moves between
/// samples, `board` stands at the last sample. The last sample's move is not part of the
/// data (the terminal position is never a sample), so its flags, and the threefold bit
/// of the game, are read off the result: every legal move that ends the game with the
/// recorded winner is a candidate, and a bit is set only when all candidates agree.
/// Self-play and the replay tool both go through here, so their records are identical.
///
/// `terminal` is how the game ended when the caller knows it (self-play: the gamelog
/// name's code); `None` reads it off the result too: the limit code for an undecided
/// game, the candidates' common terminal for a decided one, `TERMINAL_UNKNOWN` when
/// they disagree or there is none. The two agree whenever the replay can tell.
#[allow(clippy::too_many_arguments)]
pub fn finish_game(
    features: &[PositionFeatures],
    played: &[(u8, u8)],
    mut move_flags: Vec<u8>,
    board: &mut Board,
    last_flags: u8,
    last_value: i8,
    terminal: Option<u8>,
    out: &mut Vec<u8>,
) -> GameEnd {
    let n = features.len();
    debug_assert_eq!(played.len() + 1, n);
    debug_assert_eq!(move_flags.len(), n);
    let last = n - 1;
    move_flags[last] |= FEAT_LAST_PLY;
    let mut end = GameEnd::default();
    let mut inferred = TERMINAL_LIMIT;
    if last_flags & FLAG_RESULT_UNKNOWN == 0 {
        let candidates = resolve_last_move(board, last_flags, last_value);
        inferred = match candidates.first() {
            Some(c) if candidates.iter().all(|x| x.terminal == c.terminal) => c.terminal,
            _ => TERMINAL_UNKNOWN,
        };
        if candidates.is_empty() {
            end.unresolved = true;
        } else {
            let all = |f: fn(&LastMove) -> bool| candidates.iter().all(f);
            let any = |f: fn(&LastMove) -> bool| candidates.iter().any(f);
            if all(|c| c.king) {
                move_flags[last] |= FEAT_KING_MOVE;
            }
            if all(|c| c.capture) {
                move_flags[last] |= FEAT_CAPTURE;
            }
            end.threefold = all(|c| c.threefold);
            let agree = |f: fn(&LastMove) -> bool| all(f) || !any(f);
            end.ambiguous = !(agree(|c| c.king) && agree(|c| c.capture) && agree(|c| c.threefold));
        }
    }
    end.terminal = terminal.unwrap_or(inferred);
    for i in 0..n {
        let mut hist = [(NO_SQUARE, NO_SQUARE); 4];
        for (k, h) in hist.iter_mut().enumerate() {
            if i > k {
                *h = played[i - 1 - k];
            }
        }
        if end.threefold {
            move_flags[i] |= FEAT_THREEFOLD;
        }
        let term = if i == last { end.terminal } else { 0 };
        write_record(out, &features[i], hist, i, n - i, move_flags[i], term);
    }
    end.flags = move_flags;
    end
}

/// The history-dependent input of one position: what a net with extra planes reads
/// besides the `BitPosition`. Same content as a side-file record minus the
/// end-of-game fields.
#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub struct ExtraFeatures {
    /// Last four plies, most recent first, (from, to) or `NO_SQUARE` twice.
    pub hist: [(u8, u8); 4],
    pub f: PositionFeatures,
    pub ply: u16,
}

impl ExtraFeatures {
    /// From the board as it stands (its move history and repetition table). The board
    /// is restored before returning.
    pub fn from_board(board: &mut Board, mg: &mut MoveGen) -> Self {
        let mut hist = [(NO_SQUARE, NO_SQUARE); 4];
        for (h, m) in hist.iter_mut().zip(board.last_moves.iter()) {
            if let Some(m) = m {
                *h = (m.from() as u8, m.to() as u8);
            }
        }
        Self {
            hist,
            f: position_features(board, mg),
            ply: board.ply,
        }
    }

    /// From a side-file record.
    pub fn from_record(r: &[u8]) -> Self {
        assert!(r.len() >= FEAT_RECORD_BYTES, "short feature record");
        let mask = |at: usize| -> SquareMask { r[at..at + 16].try_into().unwrap() };
        let mut hist = [(NO_SQUARE, NO_SQUARE); 4];
        for (k, h) in hist.iter_mut().enumerate() {
            *h = (r[offset::HIST + 2 * k], r[offset::HIST + 2 * k + 1]);
        }
        Self {
            hist,
            f: PositionFeatures {
                rep3: mask(offset::REP3),
                rep2: mask(offset::REP2),
                king1: mask(offset::KING1),
                king2: mask(offset::KING2),
                cap_now: mask(offset::CAP_NOW),
                cap_next: mask(offset::CAP_NEXT),
                atk_mob: mask(offset::ATK_MOB),
                def_mob: mask(offset::DEF_MOB),
            },
            ply: u16::from_le_bytes([r[offset::PLY], r[offset::PLY + 1]]),
        }
    }
}

/// The side file's 16-byte header.
pub fn feature_header() -> [u8; FEAT_HEADER_BYTES] {
    let mut header = [0u8; FEAT_HEADER_BYTES];
    header[..8].copy_from_slice(FEAT_MAGIC);
    header[8..12].copy_from_slice(&(FEAT_RECORD_BYTES as u32).to_le_bytes());
    header
}
