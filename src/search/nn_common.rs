use crate::board::position_export::BitPosition;
use crate::board::{Board, Precomputed};
use crate::movegen::MoveGen;
use crate::search::features::{ExtraFeatures, NO_SQUARE, SquareMask};
use crate::utils::bfs::bfs;
use std::fmt;

/// The 11 planes every net so far reads, in channel order. Unrelated to the board
/// side — it just happens to be 11 as well.
pub const BASE_PLANES: [&str; 11] = [
    "attackers",
    "defenders",
    "king",
    "stm",
    "throne",
    "corners",
    "edges",
    "group_bfs",
    "king_bfs",
    "rep1",
    "rep2",
];

/// `-name` -> the base planes it removes, in canonical order.
const REMOVALS: [(&str, &[usize]); 7] = [
    ("-throne", &[4]),
    ("-corners", &[5]),
    ("-edges", &[6]),
    ("-stm", &[3]),
    ("-group_bfs", &[7]),
    ("-king_bfs", &[8]),
    ("-rep", &[9, 10]),
];

/// Planes appended after the kept base planes (docs/probe-input-planes.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extra {
    /// From/to of the last 1, 2 or 4 plies, most recent first.
    Last(u8),
    /// Destinations that repeat a position a third time, then a second time.
    RepDst,
    /// King destinations in one move, then in exactly two.
    King12,
    /// `min(ply, 400) / 400` everywhere.
    Ply,
    /// Opponent pieces we can capture now, our pieces they could capture next.
    Cap,
    /// Attacker mobility, defender mobility.
    Mob,
    /// All ones.
    Ones,
}

impl Extra {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "last1" => Extra::Last(1),
            "last2" => Extra::Last(2),
            "last4" => Extra::Last(4),
            "rep_dst" => Extra::RepDst,
            "king12" => Extra::King12,
            "ply" => Extra::Ply,
            "cap" => Extra::Cap,
            "mob" => Extra::Mob,
            "ones" => Extra::Ones,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Extra::Last(1) => "last1",
            Extra::Last(2) => "last2",
            Extra::Last(_) => "last4",
            Extra::RepDst => "rep_dst",
            Extra::King12 => "king12",
            Extra::Ply => "ply",
            Extra::Cap => "cap",
            Extra::Mob => "mob",
            Extra::Ones => "ones",
        }
    }

    fn channels(self) -> usize {
        match self {
            Extra::Last(k) => 2 * k as usize,
            Extra::Ply | Extra::Ones => 1,
            _ => 2,
        }
    }
}

/// The input planes of one net: the kept base planes in base order, then the extra
/// groups in the listed order. Same grammar, errors and channel order as
/// `dataset.parse_planes` on the Python side (checked against a shared fixture).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaneSet {
    base: Vec<usize>,
    extras: Vec<Extra>,
}

impl Default for PlaneSet {
    fn default() -> Self {
        Self::base()
    }
}

impl PlaneSet {
    /// The 11 planes of every net so far.
    pub fn base() -> Self {
        Self {
            base: (0..BASE_PLANES.len()).collect(),
            extras: Vec::new(),
        }
    }

    pub fn parse(spec: &str) -> Result<Self, String> {
        let names: Vec<&str> = spec.split(',').map(str::trim).collect();
        if spec.trim().is_empty() || names.iter().any(|n| n.is_empty()) {
            return Err(format!("empty plane name in {spec:?}"));
        }
        let is_removal = |n: &str| REMOVALS.iter().any(|(r, _)| *r == n);
        let mut seen: Vec<&str> = Vec::new();
        for &n in &names {
            if seen.contains(&n) {
                return Err(format!("plane {n:?} given twice in {spec:?}"));
            }
            seen.push(n);
            if n != "base" && !is_removal(n) && Extra::parse(n).is_none() {
                return Err(format!("unknown plane {n:?}"));
            }
        }
        let history: Vec<&str> = names
            .iter()
            .copied()
            .filter(|n| matches!(Extra::parse(n), Some(Extra::Last(_))))
            .collect();
        if history.len() > 1 {
            return Err(format!(
                "{} overlap: pick one of last1, last2, last4",
                history.join(" and ")
            ));
        }
        let mut kept = [names.contains(&"base"); 11];
        for &n in &names {
            if let Some((_, planes)) = REMOVALS.iter().find(|(r, _)| *r == n) {
                for &p in planes.iter() {
                    if !kept[p] {
                        return Err(format!("{n} removes a plane that is not in {spec:?}"));
                    }
                    kept[p] = false;
                }
            }
        }
        let base: Vec<usize> = (0..11).filter(|&i| kept[i]).collect();
        let extras: Vec<Extra> = names.iter().filter_map(|n| Extra::parse(n)).collect();
        if base.is_empty() && extras.is_empty() {
            return Err(format!("no planes left in {spec:?}"));
        }
        Ok(Self { base, extras })
    }

    /// Channel count.
    pub fn len(&self) -> usize {
        self.base.len() + self.extras.iter().map(|e| e.channels()).sum::<usize>()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_base(&self) -> bool {
        self.base.len() == BASE_PLANES.len() && self.extras.is_empty()
    }

    /// Whether filling the input needs `ExtraFeatures` (anything but `ones`).
    pub fn needs_features(&self) -> bool {
        self.extras.iter().any(|e| *e != Extra::Ones)
    }

    /// Floats per sample on a board of this side.
    pub fn sample_size(&self, board_size: usize) -> usize {
        self.len() * board_size * board_size
    }

    /// One name per channel, in channel order (the shared fixture compares these).
    pub fn channel_names(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .base
            .iter()
            .map(|&i| BASE_PLANES[i].to_string())
            .collect();
        for e in &self.extras {
            match e {
                Extra::Last(k) => {
                    for i in 0..*k {
                        out.push(format!("hist{i}_from"));
                        out.push(format!("hist{i}_to"));
                    }
                }
                Extra::RepDst => out.extend(["rep3".into(), "rep2".into()]),
                Extra::King12 => out.extend(["king1".into(), "king2".into()]),
                Extra::Ply => out.push("ply".into()),
                Extra::Cap => out.extend(["cap_now".into(), "cap_next".into()]),
                Extra::Mob => out.extend(["atk_mob".into(), "def_mob".into()]),
                Extra::Ones => out.push("ones".into()),
            }
        }
        out
    }
}

/// Canonical form: `base` for the 11 of today; otherwise `base` (if any base plane is
/// kept) and its removals in fixed order, then the extras in the listed order.
impl fmt::Display for PlaneSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts: Vec<&str> = Vec::new();
        if !self.base.is_empty() {
            parts.push("base");
            for (name, planes) in REMOVALS.iter() {
                if planes.iter().all(|p| !self.base.contains(p)) {
                    parts.push(name);
                }
            }
        }
        parts.extend(self.extras.iter().map(|e| e.name()));
        write!(f, "{}", parts.join(","))
    }
}

/// What a net is fed for one position: the sample's `BitPosition` and, for nets with
/// extra planes, the history-dependent features (`None` for base-only nets, which then
/// pay nothing for them).
#[derive(Clone)]
pub struct NnInput {
    pub bp: BitPosition,
    pub extra: Option<Box<ExtraFeatures>>,
}

impl NnInput {
    /// Input with the base planes only.
    pub fn base(bp: BitPosition) -> Self {
        Self { bp, extra: None }
    }

    /// The position on `board`, with the extra features when `planes` needs them. The
    /// board is restored before returning.
    pub fn from_board(board: &mut Board, mg: &mut MoveGen, planes: &PlaneSet) -> Self {
        let rep = board.rep_table.get(&board.zobrist).copied().unwrap_or(1);
        let bp = BitPosition::from_board(board, rep);
        let extra = planes
            .needs_features()
            .then(|| Box::new(ExtraFeatures::from_board(board, mg)));
        Self { bp, extra }
    }
}

/// Fills one sample's input planes (`planes.sample_size` floats, zeroed by the caller).
pub fn fill_input(input: &mut [f32], pos: &NnInput, planes: &PlaneSet, geom: &Precomputed) {
    let sqs = geom.sqs;
    debug_assert!(input.len() == planes.sample_size(geom.board_size));
    if planes.is_base() {
        fill_base_planes(input, &pos.bp, geom);
        return;
    }
    if !planes.base.is_empty() {
        let mut all = vec![0.0f32; BASE_PLANES.len() * sqs];
        fill_base_planes(&mut all, &pos.bp, geom);
        for (c, &p) in planes.base.iter().enumerate() {
            input[c * sqs..(c + 1) * sqs].copy_from_slice(&all[p * sqs..(p + 1) * sqs]);
        }
    }
    let mut c = planes.base.len();
    let extra = pos.extra.as_deref();
    let need = || extra.expect("this net's planes need ExtraFeatures (NnInput::from_board)");
    // Channel `c` gets a square mask, or (`None`) the constant `v` everywhere.
    let mut put = |c: &mut usize, m: Option<&SquareMask>, v: f32| {
        let out = &mut input[*c * sqs..(*c + 1) * sqs];
        match m {
            Some(m) => {
                for (sq, x) in out.iter_mut().enumerate() {
                    if (m[sq / 8] >> (sq % 8)) & 1 == 1 {
                        *x = 1.0;
                    }
                }
            }
            None => out.fill(v),
        }
        *c += 1;
    };
    for e in &planes.extras {
        match *e {
            Extra::Last(k) => {
                let x = need();
                for &(from, to) in x.hist.iter().take(k as usize) {
                    let mut m = SquareMask::default();
                    if from != NO_SQUARE {
                        m[from as usize / 8] |= 1 << (from % 8);
                    }
                    put(&mut c, Some(&m), 0.0);
                    let mut m = SquareMask::default();
                    if to != NO_SQUARE {
                        m[to as usize / 8] |= 1 << (to % 8);
                    }
                    put(&mut c, Some(&m), 0.0);
                }
            }
            Extra::RepDst => {
                let x = need();
                put(&mut c, Some(&x.f.rep3), 0.0);
                put(&mut c, Some(&x.f.rep2), 0.0);
            }
            Extra::King12 => {
                let x = need();
                put(&mut c, Some(&x.f.king1), 0.0);
                put(&mut c, Some(&x.f.king2), 0.0);
            }
            Extra::Cap => {
                let x = need();
                put(&mut c, Some(&x.f.cap_now), 0.0);
                put(&mut c, Some(&x.f.cap_next), 0.0);
            }
            Extra::Mob => {
                let x = need();
                put(&mut c, Some(&x.f.atk_mob), 0.0);
                put(&mut c, Some(&x.f.def_mob), 0.0);
            }
            Extra::Ply | Extra::Ones => {
                // Same arithmetic as dataset.extra_planes (a float64 division rounded
                // to float32), so both sides agree bit for bit.
                let v = if *e == Extra::Ones {
                    1.0f32
                } else {
                    (need().ply.min(400) as f64 / 400.0) as f32
                };
                put(&mut c, None, v);
            }
        }
    }
    debug_assert_eq!(c, planes.len());
}

pub fn build_input_data(positions: &[&NnInput], planes: &PlaneSet, geom: &Precomputed) -> Vec<f32> {
    let sample_size = planes.sample_size(geom.board_size);
    let mut input_data = vec![0.0f32; positions.len() * sample_size];
    for (i, pos) in positions.iter().enumerate() {
        let start = i * sample_size;
        fill_input(
            &mut input_data[start..start + sample_size],
            pos,
            planes,
            geom,
        );
    }
    input_data
}

pub struct NnOutput {
    /// One logit per action, `get_policy_size(board_size)` long.
    pub policy: Vec<f32>,
    pub value: f32,
    /// Predicted plies-to-game-end (in units of plies / MOVES_LEFT_SCALE=100),
    /// from a moves-left head. `None` for nets without the head — the engine
    /// then simply skips the moves-left search utility (backward compatible).
    pub moves_left: Option<f32>,
}

/// The 11 base planes, exactly as every net so far was trained on them.
fn fill_base_planes(input: &mut [f32], pos: &BitPosition, geom: &Precomputed) {
    let sqs = geom.sqs;
    debug_assert!(input.len() == BASE_PLANES.len() * sqs);

    // Planes 0-2: unpack bit planes (attackers, defenders, king)
    for plane in 0..3 {
        let base = plane * 16;
        let out_offset = plane * sqs;
        for idx in 0..sqs {
            let byte = idx / 8;
            let bit = idx % 8;
            if (pos.planes[base + byte] >> bit) & 1 == 1 {
                input[out_offset + idx] = 1.0;
            }
        }
    }

    // Plane 3: side to move (all 1s if defenders, stm == 1)
    let stm_val = if pos.stm == 1 { 1.0f32 } else { 0.0f32 };
    let stm_offset = 3 * sqs;
    for i in 0..sqs {
        input[stm_offset + i] = stm_val;
    }

    // Plane 4: throne
    let throne_offset = 4 * sqs;
    input[throne_offset + geom.throne_sq] = 1.0;

    // Plane 5: corners
    let corners_offset = 5 * sqs;
    for &sq in &geom.corners_sq {
        input[corners_offset + sq] = 1.0;
    }

    // Plane 6: edge squares
    let edges_offset = 6 * sqs;
    for &sq in &geom.edges_sq {
        input[edges_offset + sq] = 1.0;
    }

    // Reconstruct piece masks from bit planes for BFS
    let mut king_sq: Option<usize> = None;
    let mut defender_seeds: Vec<usize> = Vec::new();
    let mut group_seeds: Vec<usize> = Vec::new();

    for idx in 0..sqs {
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
            &geom.vertical_horizontal_neighbors,
            &group_seeds,
        )
    } else {
        vec![false; sqs]
    };
    let group_offset = 7 * sqs;
    for idx in 0..sqs {
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
            &geom.vertical_horizontal_neighbors,
            &[ksq],
        )
    } else {
        vec![false; sqs]
    };
    let king_offset = 8 * sqs;
    // Include king square itself in king BFS plane
    if let Some(ksq) = king_sq {
        input[king_offset + ksq] = 1.0;
    }
    for idx in 0..sqs {
        if king_reach[idx] {
            input[king_offset + idx] = 1.0;
        }
    }

    // Plane 9: repetition x1 — position seen at least once before (rep >= 2)
    let rep1_val = if pos.rep >= 2 { 1.0f32 } else { 0.0 };
    let rep1_offset = 9 * sqs;
    for i in 0..sqs {
        input[rep1_offset + i] = rep1_val;
    }

    // Plane 10: repetition x2 — position seen at least twice before (rep >= 3)
    let rep2_val = if pos.rep >= 3 { 1.0f32 } else { 0.0 };
    let rep2_offset = 10 * sqs;
    for i in 0..sqs {
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
        get_square_from_algebraic(alg, 11)
    }

    fn make_input(board: &Board, rep: u8) -> Vec<f32> {
        let bp = BitPosition::from_board(board, rep);
        let planes = PlaneSet::base();
        let mut input = vec![0.0f32; planes.sample_size(board.board_size())];
        fill_input(&mut input, &NnInput::base(bp), &planes, board.precomputed());
        input
    }

    fn pv(input: &[f32], plane: usize, square: usize, board: &Board) -> f32 {
        input[plane * board.sqs() + square]
    }

    #[test]
    fn piece_planes() {
        let mut board = Board::new();
        board.set_piece(sq("a1"), Piece::ATTACKER).unwrap();
        board.set_piece(sq("b2"), Piece::DEFENDER).unwrap();
        board.set_piece(sq("c3"), Piece::KING).unwrap();
        let input = make_input(&board, 1);

        assert_eq!(pv(&input, 0, sq("a1"), &board), 1.0);
        assert_eq!(pv(&input, 0, sq("b2"), &board), 0.0);
        assert_eq!(pv(&input, 1, sq("b2"), &board), 1.0);
        assert_eq!(pv(&input, 1, sq("a1"), &board), 0.0);
        assert_eq!(pv(&input, 2, sq("c3"), &board), 1.0);
        assert_eq!(pv(&input, 2, sq("a1"), &board), 0.0);
    }

    #[test]
    fn stm_plane() {
        let mut board = Board::new();
        board.set_piece(sq("b2"), Piece::KING).unwrap();

        // Attackers to move (default, stm=0) → plane 3 all 0
        let input = make_input(&board, 1);
        assert_eq!(pv(&input, 3, sq("a1"), &board), 0.0);
        assert_eq!(pv(&input, 3, sq("k11"), &board), 0.0);

        // Defenders to move (stm=1) → plane 3 all 1
        board.side_to_move = Side::DEFENDERS;
        let input = make_input(&board, 1);
        assert_eq!(pv(&input, 3, sq("a1"), &board), 1.0);
        assert_eq!(pv(&input, 3, sq("k11"), &board), 1.0);
    }

    #[test]
    fn throne_plane() {
        let board = Board::new();
        let input = make_input(&board, 1);
        assert_eq!(pv(&input, 4, board.precomputed().throne_sq, &board), 1.0);
        assert_eq!(pv(&input, 4, sq("a1"), &board), 0.0);
    }

    #[test]
    fn corners_plane() {
        let board = Board::new();
        let input = make_input(&board, 1);
        for &csq in &board.precomputed().corners_sq {
            assert_eq!(pv(&input, 5, csq, &board), 1.0);
        }
        assert_eq!(pv(&input, 5, sq("b1"), &board), 0.0);
    }

    #[test]
    fn edges_plane() {
        let board = Board::new();
        let input = make_input(&board, 1);
        for &esq in &board.precomputed().edges_sq {
            assert_eq!(pv(&input, 6, esq, &board), 1.0);
        }
        assert_eq!(pv(&input, 6, board.precomputed().throne_sq, &board), 0.0);
    }

    // Validates the bug fix: before the fix, king BFS only lit up the king's own
    // square (same as plane 2). After the fix it expands to all reachable empty squares.
    #[test]
    fn king_bfs_reaches_neighbors() {
        let mut board = Board::new();
        board.set_piece(sq("d4"), Piece::KING).unwrap();
        let input = make_input(&board, 1);

        assert_eq!(pv(&input, 8, sq("d4"), &board), 1.0); // king's own square
        assert_eq!(pv(&input, 8, sq("c4"), &board), 1.0); // left
        assert_eq!(pv(&input, 8, sq("e4"), &board), 1.0); // right
        assert_eq!(pv(&input, 8, sq("d3"), &board), 1.0); // down
        assert_eq!(pv(&input, 8, sq("d5"), &board), 1.0); // up
        assert_eq!(pv(&input, 8, sq("a1"), &board), 1.0); // far corner reachable on empty board
    }

    #[test]
    fn king_bfs_blocked_by_attacker() {
        let mut board = Board::new();
        board.set_piece(sq("d4"), Piece::KING).unwrap();
        board.set_piece(sq("a4"), Piece::ATTACKER).unwrap();
        let input = make_input(&board, 1);

        assert_eq!(pv(&input, 8, sq("b4"), &board), 1.0); // reachable (between king and attacker)
        assert_eq!(pv(&input, 8, sq("a4"), &board), 0.0); // attacker itself not passable
        assert_eq!(pv(&input, 8, sq("k4"), &board), 1.0); // other direction still open
    }

    #[test]
    fn king_bfs_blocked_by_defender() {
        let mut board = Board::new();
        board.set_piece(sq("d4"), Piece::KING).unwrap();
        board.set_piece(sq("a4"), Piece::DEFENDER).unwrap();
        let input = make_input(&board, 1);

        assert_eq!(pv(&input, 8, sq("b4"), &board), 1.0); // reachable
        assert_eq!(pv(&input, 8, sq("a4"), &board), 0.0); // defender blocks king sliding
    }

    #[test]
    fn group_bfs_expands_through_defenders() {
        let mut board = Board::new();
        board.set_piece(sq("a1"), Piece::KING).unwrap();
        board.set_piece(sq("a2"), Piece::DEFENDER).unwrap();
        board.set_piece(sq("b1"), Piece::ATTACKER).unwrap();
        let input = make_input(&board, 1);

        assert_eq!(pv(&input, 7, sq("a1"), &board), 1.0); // king — passable (not attacker)
        assert_eq!(pv(&input, 7, sq("a2"), &board), 1.0); // defender — passable in group BFS
        assert_eq!(pv(&input, 7, sq("a3"), &board), 1.0); // reachable past defender
        assert_eq!(pv(&input, 7, sq("b1"), &board), 0.0); // attacker blocks
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

        assert_eq!(pv(&input, 7, sq("c3"), &board), 1.0); // king reachable
        // Squares below the attacker wall are cut off from king
        assert_eq!(pv(&input, 7, sq("c1"), &board), 0.0);
    }

    #[test]
    fn rep_planes() {
        let mut board = Board::new();
        board.set_piece(sq("b2"), Piece::KING).unwrap();

        let i1 = make_input(&board, 1);
        assert_eq!(pv(&i1, 9, 0, &board), 0.0);
        assert_eq!(pv(&i1, 10, 0, &board), 0.0);

        let i2 = make_input(&board, 2);
        assert_eq!(pv(&i2, 9, 0, &board), 1.0);
        assert_eq!(pv(&i2, 10, 0, &board), 0.0);

        let i3 = make_input(&board, 3);
        assert_eq!(pv(&i3, 9, 0, &board), 1.0);
        assert_eq!(pv(&i3, 10, 0, &board), 1.0);
    }
}

/// The fixtures shared with the Python side (zero-trainer/testdata, written by
/// test_planes_shared.py): plane specs with their canonical form and channel order, and
/// input tensors Python built from the two-game feature fixture.
#[cfg(test)]
mod shared_fixture_tests {
    use super::*;
    use crate::board::rules::RulesEnum;
    use crate::search::features::FEAT_RECORD_BYTES;
    use std::path::Path;

    fn testdata() -> &'static Path {
        Path::new("zero-trainer/testdata")
    }

    fn fixture() -> serde_json::Value {
        let text = std::fs::read_to_string(testdata().join("planes.json")).unwrap();
        serde_json::from_str(&text).unwrap()
    }

    #[test]
    fn planes_fixture_specs_parse_to_the_same_channels() {
        let fx = fixture();
        let specs = fx["specs"].as_array().unwrap();
        assert!(specs.len() >= 10);
        for item in specs {
            let spec = item["spec"].as_str().unwrap();
            let ps = PlaneSet::parse(spec).unwrap_or_else(|e| panic!("{spec}: {e}"));
            assert_eq!(
                ps.to_string(),
                item["canonical"].as_str().unwrap(),
                "{spec}"
            );
            let channels: Vec<String> = item["channels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c.as_str().unwrap().to_string())
                .collect();
            assert_eq!(ps.channel_names(), channels, "{spec}");
            assert_eq!(ps.len(), channels.len());
            assert_eq!(PlaneSet::parse(&ps.to_string()).unwrap(), ps, "{spec}");
        }
        assert!(PlaneSet::parse("base").unwrap().is_base());
        assert_eq!(PlaneSet::base().to_string(), "base");
    }

    // The same rejections as dataset.parse_planes (test_planes.py::test_plane_list_rejects).
    #[test]
    fn plane_spec_rejects_what_python_rejects() {
        for bad in [
            "base,base",
            "base,last2,last2",
            "base,-throne,-throne",
            "base,last1,last2",
            "base,history",
            "base,-king",
            "-throne",
            "",
            "base,",
            "base,-rep,-rep",
        ] {
            assert!(PlaneSet::parse(bad).is_err(), "{bad:?} accepted");
        }
    }

    // Python stacked these inputs with dataset.stack_planes from features.bin/.feat; the
    // engine fills them from the same samples and records.
    #[test]
    fn planes_fixture_inputs_match_python() {
        let bin = std::fs::read(testdata().join("features.bin")).unwrap();
        let feat = std::fs::read(testdata().join("features.feat")).unwrap();
        let mask_bytes = crate::mcts::export::legal_mask_bytes(11);
        let mut samples = Vec::new();
        let mut at = 0;
        while at < bin.len() {
            let mut planes = [0u8; 48];
            planes.copy_from_slice(&bin[at..at + 48]);
            let bp = BitPosition {
                planes,
                stm: bin[at + 48],
                rep: bin[at + 49],
            };
            let plen_at = at + 50 + mask_bytes;
            let plen = u16::from_le_bytes([bin[plen_at], bin[plen_at + 1]]) as usize;
            at = plen_at + 2 + plen * 4 + 4;
            samples.push(bp);
        }
        let records: Vec<&[u8]> = feat[16..].chunks(FEAT_RECORD_BYTES).collect();
        assert_eq!(samples.len(), records.len());
        let mut board = Board::new();
        board.set_rules(RulesEnum::Copenhagen11x11);
        let geom = board.precomputed();

        let fx = fixture();
        for item in fx["inputs"].as_array().unwrap() {
            let spec = item["spec"].as_str().unwrap();
            let planes = PlaneSet::parse(spec).unwrap();
            let raw = std::fs::read(testdata().join(item["file"].as_str().unwrap())).unwrap();
            let want: Vec<f32> = raw
                .chunks(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let inputs: Vec<NnInput> = samples
                .iter()
                .zip(&records)
                .map(|(bp, r)| NnInput {
                    bp: *bp,
                    extra: Some(Box::new(ExtraFeatures::from_record(r))),
                })
                .collect();
            let refs: Vec<&NnInput> = inputs.iter().collect();
            let got = build_input_data(&refs, &planes, geom);
            assert_eq!(got.len(), want.len(), "{spec}");
            let diff = got
                .iter()
                .zip(&want)
                .position(|(a, b)| a.to_bits() != b.to_bits());
            assert_eq!(diff, None, "{spec}: first differing float");
        }
    }

    // A replayed game's board history gives the same ExtraFeatures as the side file.
    #[test]
    fn extra_features_from_board_match_the_records() {
        let feat = std::fs::read(testdata().join("features.feat")).unwrap();
        let records: Vec<&[u8]> = feat[16..].chunks(FEAT_RECORD_BYTES).collect();
        // The fixture's second game: start position, then this cycle (see feature_dump).
        let cycle = ["d1c1", "f4e4", "c1d1", "e4f4"];
        let moves: Vec<&str> = cycle
            .iter()
            .chain(cycle.iter())
            .chain(cycle[..1].iter())
            .copied()
            .collect();
        let mut board = Board::new();
        board.set_rules(RulesEnum::Copenhagen11x11);
        board.setup_initial_position().unwrap();
        let mut mg = MoveGen::new();
        for (i, m) in moves.iter().enumerate() {
            let x = ExtraFeatures::from_board(&mut board, &mut mg);
            assert_eq!(x, ExtraFeatures::from_record(records[3 + i]), "sample {i}");
            let mv = board.create_move_from_algebraic(m).unwrap();
            board.make_move_simple(mv).unwrap();
        }
    }
}
