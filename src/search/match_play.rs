//! Paired match between two nets inside one process (`taflzero --match`).
//!
//! `tools/match.py` used to drive two UCI engines per pair, each searching alone with
//! 8-leaf batches: a 30-pair match was thousands of tiny GPU calls a second next to
//! the self-play worker's big ones. Here every game of the match is a thread, each
//! search asks for one leaf at a time, and the two batch servers (one per net) pool
//! the leaves of all games that are on that side's move into one call. The games are
//! the same games a UCI engine plays: `MCTSConfig::default_play` at the fixed node
//! budget (its 8-leaf batches too, unless `search_batch` says otherwise: a game is a
//! chain of `nodes * plies / search_batch` net calls, so this bounds the wall time of
//! the whole match), one tree per side per game reused between moves, the opening
//! played by random legal moves and then from both sides.

use crate::board::Board;
use crate::board::position_export::BitPosition;
use crate::board::rules::RulesEnum;
use crate::board::types::{Piece, Side};
use crate::board::utils::get_side_by_piece;
use crate::mcts::{MCTSConfig, MCTSTree, get_left_moves, mcts_search};
use crate::movegen::MoveGen;
use crate::mv::Move;
use crate::search::nn::{BatchServer, NeuralNet};
use crate::search_data::SearchData;
use crate::terminal::{check_terminal, get_terminal};
use rand::prelude::*;
use std::hash::{BuildHasher, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone)]
pub struct MatchConfig {
    pub pairs: usize,
    pub nodes: u64,
    /// Random legal moves before the real game (as in tools/match.py: a quarter of
    /// the openings are one move shorter, so they are not all the same length).
    pub opening_moves: usize,
    /// Searched moves after the opening before the game is a draw.
    pub max_plies: usize,
    /// Game threads; every game of the match at once when it covers `2 * pairs`.
    pub threads: usize,
    /// Leaves per net call inside one search (UCI play uses 8).
    pub search_batch: usize,
    pub variant: RulesEnum,
    /// Seeds the openings; random when `None`.
    pub seed: Option<u64>,
}

/// One finished game. `winner` is `attackers` / `defenders` / `draw`, `reason` is the
/// terminal's name (`def_fort`, `atk_capture`, ... `max_plies`), `plies` the total
/// length with the opening.
#[derive(Clone, Debug, PartialEq)]
pub struct GameOutcome {
    pub pair: usize,
    pub a_attacks: bool,
    pub winner: &'static str,
    pub reason: &'static str,
    pub plies: usize,
    /// Every move of the game: the opening first, then the searched ones.
    pub moves: Vec<Move>,
    /// One per searched move: the root value of that search from the mover's view
    /// (`MCTSTree::root_q`). Opening moves are random and have none.
    pub evals: Vec<f32>,
}

/// What `play_game` hands back: winner, reason, and the record of the game.
struct PlayedGame {
    winner: &'static str,
    reason: &'static str,
    moves: Vec<Move>,
    evals: Vec<f32>,
}

fn start_board(variant: RulesEnum) -> Board {
    let mut b = Board::new();
    b.set_rules(variant);
    b.setup_initial_position()
        .expect("Setup initial position failed");
    b
}

/// `taflzero --replay`: the positions of a recorded game, the start position first and
/// then one after each move of `moves` (space-separated, the notation of the `moves`
/// line). The dashboard's game viewer has no board of its own in Python, so the engine
/// replays match games for it. `rep` is the board's own repetition count, as self-play
/// writes it. Errors name the move that does not parse or cannot be made.
pub fn replay_positions(variant: RulesEnum, moves: &str) -> Result<Vec<BitPosition>, String> {
    let mut board = start_board(variant);
    let snapshot = |b: &Board| {
        let rep = b.rep_table.get(&b.zobrist).copied().unwrap_or(1);
        BitPosition::from_board(b, rep)
    };
    let mut out = vec![snapshot(&board)];
    for (i, raw) in moves.split_whitespace().enumerate() {
        let mv = board
            .create_move_from_algebraic(raw)
            .map_err(|e| format!("move {} '{raw}': {e}", i + 1))?;
        // `make_move` trusts its input; a record that drifted from the board (wrong
        // variant, truncated line) would otherwise replay into nonsense silently.
        let mover = get_side_by_piece(board.board[mv.from()]).ok();
        if mover != Some(board.side_to_move) || board.board[mv.to()] != Piece::EMPTY {
            return Err(format!(
                "move {} '{raw}': not a move of the side to move",
                i + 1
            ));
        }
        board
            .make_move_simple(mv)
            .map_err(|e| format!("move {} '{raw}': {e}", i + 1))?;
        out.push(snapshot(&board));
    }
    Ok(out)
}

/// The opening of pair `pair`: a deterministic function of the seed, so both games of
/// the pair start the same whichever threads play them.
pub fn make_opening(variant: RulesEnum, moves: usize, seed: u64, pair: usize) -> Vec<Move> {
    let mut rng = StdRng::seed_from_u64(seed ^ (pair as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut board = start_board(variant);
    let mut mg = MoveGen::new();
    let mut out = Vec::with_capacity(moves);
    for _ in 0..moves {
        if check_terminal(&mut board).is_some() {
            break;
        }
        let legal = get_left_moves(&board, &mut mg);
        if legal.is_empty() {
            break;
        }
        let mv = legal[rng.random_range(0..legal.len())];
        board.make_move_simple(mv).expect("Make move failed");
        out.push(mv);
    }
    if !out.is_empty() && rng.random::<f64>() < 0.25 {
        out.pop();
    }
    out
}

/// Who has won in `board` and how, counting a side with no legal move as lost.
fn result_reason(board: &mut Board, mg: &mut MoveGen) -> Option<(&'static str, &'static str)> {
    if let Some(side) = check_terminal(board) {
        let reason = get_terminal(board).map_or("terminal", |t| t.name());
        return Some((side_name(side), reason));
    }
    if get_left_moves(board, mg).is_empty() {
        let side = Side::opposite(board.side_to_move);
        let reason = if side == Side::ATTACKERS {
            "atk_no_moves"
        } else {
            "def_no_moves"
        };
        return Some((side_name(side), reason));
    }
    None
}

fn side_name(side: Side) -> &'static str {
    if side == Side::ATTACKERS {
        "attackers"
    } else {
        "defenders"
    }
}

/// One game: `nn_atk` moves for the attackers, `nn_def` for the defenders.
fn play_game(
    nn_atk: &mut NeuralNet,
    nn_def: &mut NeuralNet,
    opening: &[Move],
    cfg: &MatchConfig,
) -> PlayedGame {
    let mut board = start_board(cfg.variant);
    for &mv in opening {
        board.make_move_simple(mv).expect("Opening move failed");
    }
    let mut moves = opening.to_vec();
    let mut evals = Vec::with_capacity(cfg.max_plies);
    let done = |winner, reason, moves, evals| PlayedGame {
        winner,
        reason,
        moves,
        evals,
    };
    let mut mg = MoveGen::new();
    let mut trees = (MCTSTree::new(), MCTSTree::new());
    let mut sd = SearchData::new();
    let mut config = MCTSConfig::default_play();
    config.batch_size = cfg.search_batch.max(1);
    for _ in 0..cfg.max_plies {
        if let Some((who, reason)) = result_reason(&mut board, &mut mg) {
            return done(who, reason, moves, evals);
        }
        let attackers = board.side_to_move == Side::ATTACKERS;
        let (nn, tree) = if attackers {
            (&mut *nn_atk, &mut trees.0)
        } else {
            (&mut *nn_def, &mut trees.1)
        };
        let Some(mv) = mcts_search(
            &mut board,
            tree,
            nn,
            &mut sd,
            None,
            Some(cfg.nodes),
            &config,
            None,
        ) else {
            return done("draw", "no_bestmove", moves, evals);
        };
        // The tree's root is still the searched position until the next reroot.
        evals.push(tree.root_q());
        board.make_move_simple(mv).expect("Make move failed");
        moves.push(mv);
    }
    done("draw", "max_plies", moves, evals)
}

/// Plays the match; `on_game` is called from the game threads as games finish, in
/// no particular order. Game `2 * pair` has `a` attacking, `2 * pair + 1` has `b`.
pub fn play_match(
    a: &Arc<BatchServer>,
    b: &Arc<BatchServer>,
    cfg: &MatchConfig,
    on_game: &(dyn Fn(GameOutcome) + Sync),
) {
    let seed = cfg
        .seed
        .unwrap_or_else(|| std::hash::RandomState::new().build_hasher().finish());
    let next = AtomicUsize::new(0);
    let games = cfg.pairs * 2;
    let threads = cfg.threads.clamp(1, games.max(1));
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let next = &next;
            scope.spawn(move || {
                let mut nn_a = NeuralNet::shared(Arc::clone(a));
                let mut nn_b = NeuralNet::shared(Arc::clone(b));
                a.client_started();
                b.client_started();
                loop {
                    let g = next.fetch_add(1, Ordering::Relaxed);
                    if g >= games {
                        break;
                    }
                    let (pair, a_attacks) = (g / 2, g % 2 == 0);
                    let opening = make_opening(cfg.variant, cfg.opening_moves, seed, pair);
                    let game = if a_attacks {
                        play_game(&mut nn_a, &mut nn_b, &opening, cfg)
                    } else {
                        play_game(&mut nn_b, &mut nn_a, &opening, cfg)
                    };
                    on_game(GameOutcome {
                        pair,
                        a_attacks,
                        winner: game.winner,
                        reason: game.reason,
                        plies: game.moves.len(),
                        moves: game.moves,
                        evals: game.evals,
                    });
                }
                a.client_stopped();
                b.client_stopped();
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::time::Duration;

    fn server(threads: usize) -> Arc<BatchServer> {
        BatchServer::start(
            "copenhagen11x11.onnx",
            11,
            threads,
            Duration::from_millis(1),
            false,
        )
        .unwrap()
    }

    // Openings are a function of the seed and the pair: both games of a pair start
    // alike, different pairs (almost surely) differ, and the moves are legal.
    #[test]
    fn openings_are_seeded_per_pair() {
        let v = RulesEnum::Copenhagen11x11;
        let o1 = make_opening(v, 16, 7, 3);
        assert_eq!(o1, make_opening(v, 16, 7, 3));
        assert_ne!(o1, make_opening(v, 16, 7, 4));
        assert_ne!(o1, make_opening(v, 16, 8, 3));
        assert!(o1.len() >= 15 && o1.len() <= 16, "{}", o1.len());
        let mut b = start_board(v);
        for mv in o1 {
            b.make_move_simple(mv).unwrap();
        }
        assert!(make_opening(v, 0, 7, 0).is_empty());
    }

    // `--replay` of three moves: four positions (start + one per move), each with the
    // piece moved and the side to move flipped; a move the board cannot make is an error.
    #[test]
    fn replay_three_moves() {
        let v = RulesEnum::Copenhagen11x11;
        let board = start_board(v);
        let sq = |s: &str| board.get_square_from_algebraic(s);
        let has = |p: &BitPosition, plane: usize, s: &str| {
            let i = sq(s);
            (p.planes[plane * 16 + i / 8] >> (i % 8)) & 1 == 1
        };
        let got = replay_positions(v, "d1d3 e5e3 h1h3").unwrap();
        assert_eq!(got.len(), 4);
        assert_eq!(got[0].planes, BitPosition::from_board(&board, 1).planes);
        assert_eq!(
            got.iter().map(|p| p.stm).collect::<Vec<_>>(),
            vec![0, 1, 0, 1]
        );
        assert!(has(&got[1], 0, "d3") && !has(&got[1], 0, "d1"));
        assert!(has(&got[2], 1, "e3") && !has(&got[2], 1, "e5"));
        assert!(has(&got[3], 0, "h3") && !has(&got[3], 0, "h1"));
        // Earlier moves stay on the board.
        assert!(has(&got[3], 0, "d3") && has(&got[3], 1, "e3"));
        assert!(got.iter().all(|p| p.rep == 1));
        assert!(replay_positions(v, "").unwrap().len() == 1);
        // A defender move on the attackers' turn, and garbage, are rejected.
        assert!(replay_positions(v, "e5e3").is_err());
        assert!(replay_positions(v, "d1d3 zz").is_err());
    }

    // A tiny match: every game of every pair is reported once, with a valid outcome
    // and a length that respects the ply cap, from more threads than games.
    #[test]
    fn match_reports_every_game_once() {
        let cfg = MatchConfig {
            pairs: 2,
            nodes: 2,
            opening_moves: 4,
            max_plies: 5,
            threads: 8,
            search_batch: 1,
            variant: RulesEnum::Copenhagen11x11,
            seed: Some(1),
        };
        let a = server(8);
        let b = server(8);
        let got = Mutex::new(Vec::new());
        play_match(&a, &b, &cfg, &|g| got.lock().unwrap().push(g));
        let mut got = got.into_inner().unwrap();
        got.sort_by_key(|g| (g.pair, !g.a_attacks));
        assert_eq!(got.len(), 4);
        for (i, g) in got.iter().enumerate() {
            assert_eq!((g.pair, g.a_attacks), (i / 2, i % 2 == 0));
            assert!(
                ["attackers", "defenders", "draw"].contains(&g.winner),
                "{}",
                g.winner
            );
            assert!(g.plies <= 4 + 5, "{} plies", g.plies);
            if g.winner == "draw" {
                assert_eq!(g.reason, "max_plies");
            }
            // The record covers the whole game: every move, and one eval per searched
            // move (the opening is the same for both games of the pair).
            let opening = make_opening(cfg.variant, cfg.opening_moves, 1, g.pair);
            assert_eq!(g.moves.len(), g.plies);
            assert_eq!(g.evals.len(), g.plies - opening.len());
            assert_eq!(&g.moves[..opening.len()], &opening[..]);
            assert!(
                g.evals.iter().all(|v| (-1.0..=1.0).contains(v)),
                "{:?}",
                g.evals
            );
        }
        let (batches, served) = a.stats();
        assert!(served > 0 && batches > 0);
    }
}
