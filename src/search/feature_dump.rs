//! Feature side file for the input-plane probe (`taflzero --features`).
//!
//! The trainer derives the 11 planes it uses today from a sample alone. The candidate
//! planes of docs/probe-input-planes.md (last moves, "which move would repeat", king
//! reach, captures, mobility) depend on the game's history, which a sample does not
//! carry. This tool replays every game of a sample file on a real `Board` and writes
//! one fixed-size record per sample, in sample order, so the Python side can memory-map
//! the records and index them by sample number.
//!
//! The replay must be exact: a move is recovered from the difference between two
//! consecutive samples and played with `make_move`, and the resulting position has to
//! match the next sample bit for bit. Anything else aborts the tool, because a silently
//! skipped game would shift every later record against its sample.

use crate::board::Board;
use crate::board::position_export::BitPosition;
use crate::board::rules::RulesEnum;
use crate::board::types::{Piece, Side, Square};
use crate::mcts::export::{FLAG_LAST_OF_GAME, FLAG_RESULT_UNKNOWN, legal_mask_bytes};
use crate::movegen::MoveGen;
use crate::mv::Move;
use crate::terminal::{TerminalType, check_terminal, get_terminal};
use crate::undo::UndoMove;
use std::fs::File;
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

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
    pub const CAP_NOW: usize = 80;
    pub const CAP_NEXT: usize = 96;
    pub const ATK_MOB: usize = 112;
    pub const DEF_MOB: usize = 128;
}

/// Square set packed like `BitPosition.planes`: square `i` is bit `i % 8` of byte `i / 8`.
pub type SquareMask = [u8; 16];

#[inline]
fn set_bit(mask: &mut SquareMask, sq: Square) {
    mask[sq / 8] |= 1 << (sq % 8);
}

#[inline]
fn has_bit(mask: &SquareMask, sq: Square) -> bool {
    (mask[sq / 8] >> (sq % 8)) & 1 == 1
}

fn popcount(mask: &SquareMask) -> u32 {
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

/// One sample as far as the replay needs it.
#[derive(Clone, Copy)]
pub struct SampleHead {
    pub bp: BitPosition,
    pub flags: u8,
    pub value: i8,
}

/// Counters printed at the end; `merge` adds a worker's share.
#[derive(Default, Debug, Clone)]
pub struct FeatureStats {
    pub games: usize,
    pub samples: usize,
    pub records: usize,
    pub rep3_nonempty: usize,
    pub rep2_nonempty: usize,
    pub king1_squares: u64,
    pub king2_squares: u64,
    pub king_moves: usize,
    pub threefold_games: usize,
    /// Decided games whose last move could not be pinned down: no legal move from
    /// the last sample reaches the recorded result, or several do and they disagree
    /// on the king / capture / threefold bits (those bits are then left clear).
    pub last_move_unresolved: usize,
    pub last_move_ambiguous: usize,
    /// Samples after the last complete game (a file being appended to).
    pub tail_samples_skipped: usize,
}

impl FeatureStats {
    fn merge(&mut self, o: &FeatureStats) {
        self.games += o.games;
        self.samples += o.samples;
        self.records += o.records;
        self.rep3_nonempty += o.rep3_nonempty;
        self.rep2_nonempty += o.rep2_nonempty;
        self.king1_squares += o.king1_squares;
        self.king2_squares += o.king2_squares;
        self.king_moves += o.king_moves;
        self.threefold_games += o.threefold_games;
        self.last_move_unresolved += o.last_move_unresolved;
        self.last_move_ambiguous += o.last_move_ambiguous;
    }

    pub fn summary(&self) -> String {
        let share = |n: usize| {
            if self.records == 0 {
                0.0
            } else {
                100.0 * n as f64 / self.records as f64
            }
        };
        let mean = |n: u64| {
            if self.records == 0 {
                0.0
            } else {
                n as f64 / self.records as f64
            }
        };
        format!(
            "games {} | samples {} | records written {}\n\
             rep3 non-empty {:.3}% | rep2 non-empty {:.3}% | king1 mean {:.3} | king2 mean {:.3} | king move {:.2}%\n\
             threefold games {} | last move unresolved {} | ambiguous {} | tail samples skipped {}",
            self.games,
            self.samples,
            self.records,
            share(self.rep3_nonempty),
            share(self.rep2_nonempty),
            mean(self.king1_squares),
            mean(self.king2_squares),
            share(self.king_moves),
            self.threefold_games,
            self.last_move_unresolved,
            self.last_move_ambiguous,
            self.tail_samples_skipped,
        )
    }
}

fn render(bp: &BitPosition, board_size: usize) -> String {
    let mut out = String::new();
    for row in (0..board_size).rev() {
        for col in 0..board_size {
            let sq = row * board_size + col;
            let bit = |plane: usize| (bp.planes[plane * 16 + sq / 8] >> (sq % 8)) & 1 == 1;
            out.push(if bit(0) {
                'A'
            } else if bit(1) {
                'D'
            } else if bit(2) {
                'K'
            } else {
                '.'
            });
        }
        out.push('\n');
    }
    out.push_str(&format!("stm {} rep {}\n", bp.stm, bp.rep));
    out
}

fn mismatch(
    game: usize,
    ply: usize,
    prev: &BitPosition,
    this: &BitPosition,
    n: usize,
    why: &str,
) -> String {
    format!(
        "game {game}, ply {ply}: {why}\nprevious sample:\n{}this sample:\n{}",
        render(prev, n),
        render(this, n)
    )
}

/// The move between two consecutive samples: the one mover square that empties and
/// the one empty square that fills.
fn recover_move(prev: &BitPosition, this: &BitPosition, sqs: usize) -> Result<Move, String> {
    let bit = |bp: &BitPosition, plane: usize, sq: usize| {
        (bp.planes[plane * 16 + sq / 8] >> (sq % 8)) & 1 == 1
    };
    let occupied = |bp: &BitPosition, sq: usize| (0..3).any(|p| bit(bp, p, sq));
    // The king counts as a defender piece.
    let mover = |bp: &BitPosition, sq: usize| {
        if prev.stm == Side::ATTACKERS as u8 {
            bit(bp, 0, sq)
        } else {
            bit(bp, 1, sq) || bit(bp, 2, sq)
        }
    };
    let froms: Vec<usize> = (0..sqs)
        .filter(|&sq| mover(prev, sq) && !occupied(this, sq))
        .collect();
    let tos: Vec<usize> = (0..sqs)
        .filter(|&sq| !occupied(prev, sq) && mover(this, sq))
        .collect();
    if froms.len() != 1 || tos.len() != 1 {
        return Err(format!(
            "expected one from and one to square, found from {froms:?} to {tos:?}"
        ));
    }
    Ok(Move::new(froms[0], tos[0]))
}

fn current_rep(board: &Board) -> u8 {
    board.rep_table.get(&board.zobrist).copied().unwrap_or(1)
}

/// What the last move of a decided game was, as far as the result tells: every legal
/// move from the last sample that ends the game with the recorded winner.
struct LastMove {
    king: bool,
    capture: bool,
    threefold: bool,
}

fn resolve_last_move(board: &mut Board, last: &SampleHead) -> Vec<LastMove> {
    if last.flags & FLAG_RESULT_UNKNOWN != 0 || last.value == 0 {
        return Vec::new();
    }
    let stm = board.side_to_move;
    let winner = if last.value > 0 {
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
            Some(t) => (check_terminal(board) == Some(winner))
                .then_some(t == TerminalType::ThreefoldRepetition),
            // Self-play also ends a game when the side to move has no move at all.
            None => {
                mg.generate_moves(board);
                (mg.count == 0 && winner == stm).then_some(false)
            }
        };
        board.unmake_move(&mut undo).expect("unmake");
        if let Some(threefold) = hit {
            out.push(LastMove {
                king,
                capture,
                threefold,
            });
        }
    }
    out
}

fn write_record(
    out: &mut Vec<u8>,
    f: &PositionFeatures,
    hist: [(u8, u8); 4],
    ply: usize,
    plies_to_end: usize,
    flags: u8,
) {
    let mut r = [0u8; FEAT_RECORD_BYTES];
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

/// Replays one game and appends its records to `out`.
pub fn game_records(
    game: &[SampleHead],
    variant: RulesEnum,
    game_index: usize,
    out: &mut Vec<u8>,
    stats: &mut FeatureStats,
) -> Result<(), String> {
    let n = game.len();
    let mut board = game[0].bp.to_board(variant);
    let board_size = board.board_size();
    let sqs = board.sqs();
    let mut mg = MoveGen::new();
    let mut undo = UndoMove::new();
    let mut features: Vec<PositionFeatures> = Vec::with_capacity(n);
    let mut played: Vec<(u8, u8)> = Vec::with_capacity(n);
    let mut move_flags: Vec<u8> = vec![0; n];

    for i in 0..n {
        let here = BitPosition::from_board(&board, current_rep(&board));
        let s = &game[i].bp;
        if here.planes != s.planes || here.stm != s.stm || here.rep != s.rep {
            let prev = if i > 0 { &game[i - 1].bp } else { s };
            let why = if here.planes != s.planes || here.stm != s.stm {
                "replayed position differs from the sample".to_string()
            } else {
                format!(
                    "repetition count {} on replay, {} in the sample",
                    here.rep, s.rep
                )
            };
            return Err(mismatch(game_index, i, prev, s, board_size, &why));
        }
        features.push(position_features(&mut board, &mut mg));

        if i + 1 < n {
            let next = &game[i + 1].bp;
            if next.stm == s.stm {
                return Err(mismatch(
                    game_index,
                    i + 1,
                    s,
                    next,
                    board_size,
                    "side to move did not flip",
                ));
            }
            let mv = recover_move(s, next, sqs)
                .map_err(|why| mismatch(game_index, i + 1, s, next, board_size, &why))?;
            mg.generate_moves(&board);
            if !mg.moves().iter().any(|m| m.raw() == mv.raw()) {
                return Err(mismatch(
                    game_index,
                    i + 1,
                    s,
                    next,
                    board_size,
                    "the recovered move is not legal",
                ));
            }
            if board.board[mv.from()] == Piece::KING {
                move_flags[i] |= FEAT_KING_MOVE;
            }
            board
                .make_move(mv, &mut undo)
                .map_err(|e| mismatch(game_index, i + 1, s, next, board_size, e))?;
            if undo.captured_pieces_count > 0 {
                move_flags[i] |= FEAT_CAPTURE;
            }
            played.push((mv.from() as u8, mv.to() as u8));
        }
    }

    let last = n - 1;
    move_flags[last] |= FEAT_LAST_PLY;
    let mut threefold = false;
    if game[last].flags & FLAG_RESULT_UNKNOWN == 0 {
        let candidates = resolve_last_move(&mut board, &game[last]);
        if candidates.is_empty() {
            stats.last_move_unresolved += 1;
        } else {
            let all = |f: fn(&LastMove) -> bool| candidates.iter().all(f);
            let any = |f: fn(&LastMove) -> bool| candidates.iter().any(f);
            if all(|c| c.king) {
                move_flags[last] |= FEAT_KING_MOVE;
            }
            if all(|c| c.capture) {
                move_flags[last] |= FEAT_CAPTURE;
            }
            threefold = all(|c| c.threefold);
            let agree = |f: fn(&LastMove) -> bool| all(f) || !any(f);
            if !(agree(|c| c.king) && agree(|c| c.capture) && agree(|c| c.threefold)) {
                stats.last_move_ambiguous += 1;
            }
        }
    }
    if threefold {
        stats.threefold_games += 1;
    }

    for i in 0..n {
        let mut hist = [(NO_SQUARE, NO_SQUARE); 4];
        for (k, h) in hist.iter_mut().enumerate() {
            if i > k {
                *h = played[i - 1 - k];
            }
        }
        let mut flags = move_flags[i];
        if threefold {
            flags |= FEAT_THREEFOLD;
        }
        let f = &features[i];
        write_record(out, f, hist, i, n - i, flags);
        stats.rep3_nonempty += (popcount(&f.rep3) > 0) as usize;
        stats.rep2_nonempty += (popcount(&f.rep2) > 0) as usize;
        stats.king1_squares += popcount(&f.king1) as u64;
        stats.king2_squares += popcount(&f.king2) as u64;
        stats.king_moves += (flags & FEAT_KING_MOVE != 0) as usize;
    }
    stats.games += 1;
    stats.samples += n;
    stats.records += n;
    Ok(())
}

/// Reads the next sample, or `None` at the end of the file or at a truncated one.
fn read_sample<R: Read>(
    r: &mut R,
    mask_bytes: usize,
    buf: &mut Vec<u8>,
) -> Result<Option<SampleHead>, String> {
    let head = size_of::<BitPosition>() + mask_bytes + 2;
    buf.resize(head, 0);
    match r.read_exact(&mut buf[..head]) {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(format!("read failed: {e}")),
    }
    let mut planes = [0u8; 48];
    planes.copy_from_slice(&buf[..48]);
    let bp = BitPosition {
        planes,
        stm: buf[48],
        rep: buf[49],
    };
    let policy_len = u16::from_le_bytes([buf[head - 2], buf[head - 1]]) as usize;
    let rest = policy_len * 4 + 4;
    buf.resize(rest, 0);
    match r.read_exact(&mut buf[..rest]) {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(format!("read failed: {e}")),
    }
    Ok(Some(SampleHead {
        bp,
        value: buf[rest - 4] as i8,
        flags: buf[rest - 2],
    }))
}

/// Games handed to the worker threads at once.
const GAMES_PER_BATCH: usize = 2048;

fn process_batch(
    batch: &[Vec<SampleHead>],
    first_game: usize,
    variant: RulesEnum,
    threads: usize,
) -> Result<(Vec<u8>, FeatureStats), String> {
    let chunk = batch.len().div_ceil(threads.max(1)).max(1);
    let parts: Vec<Result<(Vec<u8>, FeatureStats), String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = batch
            .chunks(chunk)
            .enumerate()
            .map(|(c, games)| {
                scope.spawn(move || {
                    let mut out = Vec::new();
                    let mut stats = FeatureStats::default();
                    for (k, g) in games.iter().enumerate() {
                        game_records(g, variant, first_game + c * chunk + k, &mut out, &mut stats)?;
                    }
                    Ok((out, stats))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("feature thread panicked"))
            .collect()
    });
    let mut out = Vec::new();
    let mut stats = FeatureStats::default();
    for p in parts {
        let (bytes, s) = p?;
        out.extend_from_slice(&bytes);
        stats.merge(&s);
    }
    Ok((out, stats))
}

fn part_path(out: &Path) -> PathBuf {
    let mut s = out.as_os_str().to_owned();
    s.push(".part");
    PathBuf::from(s)
}

/// `taflzero --features`: one record per sample of every complete game in `input`
/// (whole games until at least `limit` samples when given), written to `output`.
/// The records go to `<output>.part` first and are renamed only when every game
/// replayed, so an aborted run never leaves a file that looks complete.
pub fn dump_features(
    input: &Path,
    output: &Path,
    limit: Option<usize>,
    variant: RulesEnum,
    threads: usize,
) -> Result<FeatureStats, String> {
    let file = File::open(input).map_err(|e| format!("{}: {e}", input.display()))?;
    let mut reader = BufReader::with_capacity(8 << 20, file);
    let mask_bytes = legal_mask_bytes(variant.rules().board_size);
    let part = part_path(output);
    let out_file = File::create(&part).map_err(|e| format!("{}: {e}", part.display()))?;
    let mut writer = BufWriter::with_capacity(8 << 20, out_file);

    let result = (|| -> Result<FeatureStats, String> {
        let mut header = [0u8; FEAT_HEADER_BYTES];
        header[..8].copy_from_slice(FEAT_MAGIC);
        header[8..12].copy_from_slice(&(FEAT_RECORD_BYTES as u32).to_le_bytes());
        writer.write_all(&header).map_err(|e| e.to_string())?;

        let mut stats = FeatureStats::default();
        let mut buf = Vec::new();
        let mut current: Vec<SampleHead> = Vec::new();
        let mut batch: Vec<Vec<SampleHead>> = Vec::new();
        let mut queued = 0usize;
        let mut flush =
            |batch: &mut Vec<Vec<SampleHead>>, stats: &mut FeatureStats| -> Result<(), String> {
                if batch.is_empty() {
                    return Ok(());
                }
                let (bytes, s) = process_batch(batch, stats.games, variant, threads)?;
                writer.write_all(&bytes).map_err(|e| e.to_string())?;
                stats.merge(&s);
                batch.clear();
                Ok(())
            };
        loop {
            if limit.is_some_and(|l| queued >= l) {
                break;
            }
            let Some(s) = read_sample(&mut reader, mask_bytes, &mut buf)? else {
                break;
            };
            let last = s.flags & FLAG_LAST_OF_GAME != 0;
            current.push(s);
            if last {
                queued += current.len();
                batch.push(std::mem::take(&mut current));
                if batch.len() >= GAMES_PER_BATCH {
                    flush(&mut batch, &mut stats)?;
                }
            }
        }
        flush(&mut batch, &mut stats)?;
        stats.tail_samples_skipped = current.len();
        writer.flush().map_err(|e| e.to_string())?;
        Ok(stats)
    })();

    match result {
        Ok(stats) => {
            std::fs::rename(&part, output).map_err(|e| format!("{}: {e}", output.display()))?;
            Ok(stats)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&part);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::set_board_from_str;
    use crate::mcts::export::{LegalMask, PendingSample};
    use crate::mcts::utils::move_to_policy_index;

    fn copenhagen(position: &str) -> Board {
        let mut b = Board::new();
        b.set_rules(RulesEnum::Copenhagen11x11);
        set_board_from_str(&mut b, position);
        b
    }

    fn start_board() -> Board {
        let mut b = Board::new();
        b.set_rules(RulesEnum::Copenhagen11x11);
        b.setup_initial_position().unwrap();
        b
    }

    fn sq(board: &Board, s: &str) -> Square {
        board.get_square_from_algebraic(s)
    }

    fn mask_of(squares: &[Square]) -> SquareMask {
        let mut m = SquareMask::default();
        for &s in squares {
            set_bit(&mut m, s);
        }
        m
    }

    fn squares_of(m: &SquareMask) -> Vec<Square> {
        (0..121).filter(|&s| has_bit(m, s)).collect()
    }

    fn destinations(board: &Board, side: Side) -> SquareMask {
        let mut copy = BitPosition::from_board(board, 1).to_board(RulesEnum::Copenhagen11x11);
        copy.set_side(side);
        let mut mg = MoveGen::new();
        mg.generate_moves(&copy);
        let mut m = SquareMask::default();
        for mv in mg.moves() {
            set_bit(&mut m, mv.to());
        }
        m
    }

    /// Plays `moves` from `start` and writes the game with the real sample writer:
    /// one sample per position before a move, the last sample closing the game.
    fn write_game(w: &mut Vec<u8>, start: &Board, moves: &[&str], result: Option<Side>) {
        let mut board = BitPosition::from_board(start, 1).to_board(RulesEnum::Copenhagen11x11);
        let mut samples = Vec::new();
        for m in moves {
            let rep = board.rep_table.get(&board.zobrist).copied().unwrap_or(1);
            let mut mg = MoveGen::new();
            mg.generate_moves(&board);
            let mut legal = LegalMask::new(11);
            for mv in mg.moves() {
                legal.set(move_to_policy_index(*mv, 11) as usize);
            }
            let mv = board.create_move_from_algebraic(m).unwrap();
            assert!(
                mg.moves().iter().any(|x| x.raw() == mv.raw()),
                "{m} is illegal"
            );
            let policy = vec![(move_to_policy_index(mv, 11), 10)];
            samples.push(PendingSample::from_manual(
                BitPosition::from_board(&board, rep),
                legal,
                policy,
                0,
                0,
            ));
            board.make_move_simple(mv).unwrap();
        }
        for s in samples.iter_mut() {
            s.set_value_from_result(result);
        }
        samples.last_mut().unwrap().set_last_of_game();
        for s in &samples {
            s.write_to(w).unwrap();
        }
    }

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("taflzero-feat-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    struct Rec<'a>(&'a [u8]);
    impl Rec<'_> {
        fn hist(&self, k: usize) -> (u8, u8) {
            (self.0[2 * k], self.0[2 * k + 1])
        }
        fn mask(&self, at: usize) -> SquareMask {
            self.0[at..at + 16].try_into().unwrap()
        }
        fn u16(&self, at: usize) -> u16 {
            u16::from_le_bytes([self.0[at], self.0[at + 1]])
        }
        fn flags(&self) -> u8 {
            self.0[offset::FLAGS]
        }
    }

    fn dump(bin: &[u8], name: &str) -> (Result<FeatureStats, String>, PathBuf) {
        let input = tmp(&format!("{name}.bin"));
        let output = tmp(&format!("{name}.feat"));
        let _ = std::fs::remove_file(&output);
        std::fs::write(&input, bin).unwrap();
        (
            dump_features(&input, &output, None, RulesEnum::Copenhagen11x11, 2),
            output,
        )
    }

    fn records(path: &Path) -> Vec<Vec<u8>> {
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(&bytes[..8], FEAT_MAGIC);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 144);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 0);
        assert_eq!((bytes.len() - 16) % 144, 0);
        bytes[16..].chunks(144).map(|c| c.to_vec()).collect()
    }

    // An attacker on f8 between a defender on e8 and an empty g8 a defender can reach
    // from g11: it is in cap_now when defenders move and in cap_next when attackers do.
    #[test]
    fn captures_and_mobility_on_a_hand_made_position() {
        let mut board = copenhagen(
            "......D....
             ...........
             ...........
             ....DA.....
             ...........
             ...........
             ...........
             ...........
             .K.........
             ...........
             ...........",
        );
        let attacker = sq(&board, "f8");
        assert_eq!(board.board[attacker], Piece::ATTACKER);
        let mut mg = MoveGen::new();

        board.set_side(Side::DEFENDERS);
        let before = board.get_fen();
        let f = position_features(&mut board, &mut mg);
        assert_eq!(board.get_fen(), before, "board restored");
        assert_eq!(squares_of(&f.cap_now), vec![attacker]);
        assert!(squares_of(&f.cap_next).is_empty());
        assert_eq!(f.atk_mob, destinations(&board, Side::ATTACKERS));
        assert_eq!(f.def_mob, destinations(&board, Side::DEFENDERS));

        board.set_side(Side::ATTACKERS);
        let f = position_features(&mut board, &mut mg);
        assert_eq!(squares_of(&f.cap_next), vec![attacker]);
        assert!(squares_of(&f.cap_now).is_empty());
        assert_eq!(f.atk_mob, destinations(&board, Side::ATTACKERS));
        assert_eq!(f.def_mob, destinations(&board, Side::DEFENDERS));
        // The king's reach does not depend on whose turn it is.
        assert!(!squares_of(&f.king1).is_empty());
    }

    // Start position and two legal moves: hist fills most recent first, ply counts up,
    // plies_to_end counts down to 1.
    #[test]
    fn three_ply_game_fills_history_in_order() {
        let start = start_board();
        let mut bin = Vec::new();
        write_game(&mut bin, &start, &["d1c1", "f4e4", "c1d1"], None);
        let (res, out) = dump(&bin, "three");
        let stats = res.unwrap();
        assert_eq!((stats.games, stats.records), (1, 3));
        let recs = records(&out);
        assert_eq!(recs.len(), 3);
        let (d1, c1, f4, e4) = (
            sq(&start, "d1") as u8,
            sq(&start, "c1") as u8,
            sq(&start, "f4") as u8,
            sq(&start, "e4") as u8,
        );
        let none = (NO_SQUARE, NO_SQUARE);
        let r: Vec<Rec> = recs.iter().map(|r| Rec(r)).collect();
        assert_eq!(
            [r[0].hist(0), r[0].hist(1), r[0].hist(2), r[0].hist(3)],
            [none; 4]
        );
        assert_eq!(
            [r[1].hist(0), r[1].hist(1), r[1].hist(2)],
            [(d1, c1), none, none]
        );
        assert_eq!(
            [r[2].hist(0), r[2].hist(1), r[2].hist(2), r[2].hist(3)],
            [(f4, e4), (d1, c1), none, none]
        );
        for (i, rec) in r.iter().enumerate() {
            assert_eq!(rec.u16(offset::PLY), i as u16);
            assert_eq!(rec.u16(offset::PLIES_TO_END), 3 - i as u16);
            assert_eq!(rec.flags() & FEAT_LAST_PLY != 0, i == 2);
            // Undecided game: nothing is known about the last move.
            assert_eq!(
                rec.flags() & (FEAT_KING_MOVE | FEAT_CAPTURE | FEAT_THREEFOLD),
                0
            );
        }
    }

    // An attacker and a defender walk back and forth. The game's first position never
    // enters the repetition table (as in self-play), so the cycle counted is the four
    // positions after it: the sample before the second visit of a position has the
    // returning destination in rep2, the sample before the third visit has it in rep3,
    // and that third visit ends the game as a threefold attacker win.
    #[test]
    fn back_and_forth_marks_rep2_then_rep3() {
        let start = start_board();
        let cycle = ["d1c1", "f4e4", "c1d1", "e4f4"];
        let moves: Vec<&str> = cycle
            .iter()
            .chain(cycle.iter())
            .chain(cycle[..1].iter())
            .copied()
            .collect();
        let mut bin = Vec::new();
        write_game(&mut bin, &start, &moves, Some(Side::ATTACKERS));
        let (res, out) = dump(&bin, "repeat");
        let stats = res.unwrap();
        assert_eq!(stats.threefold_games, 1);
        assert_eq!(stats.last_move_unresolved + stats.last_move_ambiguous, 0);
        let recs = records(&out);
        assert_eq!(recs.len(), 9);
        let dest = |m: &str| start.create_move_from_algebraic(m).unwrap().to();
        for (i, rec) in recs.iter().enumerate() {
            let rec = Rec(rec);
            let (rep2, rep3) = match i {
                4..=7 => (mask_of(&[dest(moves[i])]), SquareMask::default()),
                8 => (SquareMask::default(), mask_of(&[dest(moves[8])])),
                _ => (SquareMask::default(), SquareMask::default()),
            };
            assert_eq!(rec.mask(offset::REP2), rep2, "rep2 at sample {i}");
            assert_eq!(rec.mask(offset::REP3), rep3, "rep3 at sample {i}");
            assert_ne!(rec.flags() & FEAT_THREEFOLD, 0);
        }
        let last = Rec(&recs[8]);
        assert_eq!(last.flags() & (FEAT_KING_MOVE | FEAT_CAPTURE), 0);
        assert_ne!(last.flags() & FEAT_LAST_PLY, 0);
    }

    /// Squares a lone slider reaches from `from` on `board` (independent of MoveGen).
    fn slide(board: &Board, from: Square) -> Vec<Square> {
        let (r0, c0) = ((from / 11) as i32, (from % 11) as i32);
        let mut out = Vec::new();
        for (dr, dc) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (mut r, mut c) = (r0 + dr, c0 + dc);
            while (0..11).contains(&r) && (0..11).contains(&c) {
                let s = (r * 11 + c) as Square;
                if board.board[s] != Piece::EMPTY {
                    break;
                }
                out.push(s);
                r += dr;
                c += dc;
            }
        }
        out
    }

    #[test]
    fn king_reach_on_the_start_position_and_an_open_line() {
        let mut board = start_board();
        let mut mg = MoveGen::new();
        let f = position_features(&mut board, &mut mg);
        assert!(squares_of(&f.king1).is_empty(), "the king is enclosed");
        assert!(squares_of(&f.king2).is_empty());

        // Clear g6 and h6: the king slides east to i6 (j6 holds an attacker).
        board.clear_piece(sq(&board, "g6"));
        board.clear_piece(sq(&board, "h6"));
        let king = board.king_sq as Square;
        let f = position_features(&mut board, &mut mg);
        let king1 = vec![sq(&board, "g6"), sq(&board, "h6"), sq(&board, "i6")];
        assert_eq!(squares_of(&f.king1), king1);

        let mut expected = SquareMask::default();
        for &s in &king1 {
            board.clear_piece(king);
            board.set_piece(s, Piece::KING).unwrap();
            for t in slide(&board, s) {
                if t != king && !king1.contains(&t) {
                    set_bit(&mut expected, t);
                }
            }
            board.clear_piece(s);
            board.set_piece(king, Piece::KING).unwrap();
        }
        assert_eq!(f.king2, expected);
        for s in ["h5", "h2", "h10", "i1", "i11"] {
            assert!(has_bit(&f.king2, sq(&board, s)), "{s} in king2");
        }
        assert!(!has_bit(&f.king2, sq(&board, "h1")), "h1 holds an attacker");
    }

    // A sample whose position does not follow from the previous one: the tool fails
    // and leaves no output behind.
    #[test]
    fn corrupted_game_is_an_error_not_a_record() {
        let start = start_board();
        let mut good = Vec::new();
        write_game(&mut good, &start, &["d1c1", "f4e4", "c1d1"], None);
        let mut other = Vec::new();
        write_game(&mut other, &start, &["h1i1", "f4e4", "i1h1"], None);
        // Splice: samples 0-1 of the first game, sample 2 of the second.
        let size = |b: &[u8], at: usize| {
            let plen = u16::from_le_bytes([b[at + 50 + 605], b[at + 50 + 606]]) as usize;
            50 + 605 + 2 + plen * 4 + 4
        };
        let s0 = size(&good, 0);
        let s1 = size(&good, s0);
        let mut bad = good[..s0 + s1].to_vec();
        bad.extend_from_slice(&other[s0 + s1..]);

        let (res, out) = dump(&bad, "corrupt");
        let err = res.unwrap_err();
        assert!(err.contains("game 0, ply 2"), "{err}");
        assert!(!out.exists());
        assert!(!part_path(&out).exists());
    }

    // The Python tests read zero-trainer/testdata/features.{bin,feat}: the two games
    // above, written by the real sample writer and this tool. The test fails when the
    // committed files drift from what the code writes; TAFLZERO_UPDATE_FIXTURES=1
    // rewrites them.
    #[test]
    fn python_fixture_matches_the_tool() {
        let start = start_board();
        let mut bin = Vec::new();
        write_game(&mut bin, &start, &["d1c1", "f4e4", "c1d1"], None);
        let cycle = ["d1c1", "f4e4", "c1d1", "e4f4"];
        let moves: Vec<&str> = cycle
            .iter()
            .chain(cycle.iter())
            .chain(cycle[..1].iter())
            .copied()
            .collect();
        write_game(&mut bin, &start, &moves, Some(Side::ATTACKERS));
        let (res, out) = dump(&bin, "fixture");
        assert_eq!(res.unwrap().records, 12);
        let feat = std::fs::read(&out).unwrap();

        let dir = Path::new("zero-trainer").join("testdata");
        if std::env::var("TAFLZERO_UPDATE_FIXTURES").is_ok_and(|v| v == "1") {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("features.bin"), &bin).unwrap();
            std::fs::write(dir.join("features.feat"), &feat).unwrap();
        }
        assert_eq!(
            std::fs::read(dir.join("features.bin")).unwrap(),
            bin,
            "rerun with TAFLZERO_UPDATE_FIXTURES=1"
        );
        assert_eq!(
            std::fs::read(dir.join("features.feat")).unwrap(),
            feat,
            "rerun with TAFLZERO_UPDATE_FIXTURES=1"
        );
    }
}
