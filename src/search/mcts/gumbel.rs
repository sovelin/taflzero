//! Gumbel AlphaZero (Danihelka et al. 2022) on top of the shared tree: sequential
//! halving over Gumbel-perturbed root logits, policy improvement
//! `softmax(logits + sigma(completed Q))` inside the tree and as the training target.
//! Formulas follow mctx (`gumbel_muzero_policy`, `qtransform_completed_by_mix_value`).

use super::{MCTSTree, NodeId};
use rand::{Rng, RngExt};

/// Root/interior selection knobs. `considered` root actions enter sequential halving.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GumbelConfig {
    pub considered: usize,
    /// sigma(q) = (maxvisit_init + max_visits) * value_scale * q
    pub value_scale: f32,
    pub maxvisit_init: f32,
    /// Sample Gumbel noise at the root (off for deterministic play).
    pub noise: bool,
}

impl Default for GumbelConfig {
    fn default() -> Self {
        Self {
            considered: 16,
            value_scale: 0.1,
            maxvisit_init: 50.0,
            noise: true,
        }
    }
}

/// Visit count at which each of `simulations` root simulations is spent, as in
/// mctx's `get_sequence_of_considered_visits`: the `considered` best actions get one
/// visit each, then half of them get more, and so on.
pub fn halving_schedule(considered: usize, simulations: u32) -> Vec<u32> {
    if considered <= 1 {
        return (0..simulations).collect();
    }
    let log2 = (considered as f64).log2().ceil() as u32;
    let mut sequence = Vec::with_capacity(simulations as usize);
    let mut visits = vec![0u32; considered];
    let mut live = considered;
    while (sequence.len() as u32) < simulations {
        let extra = (simulations / (log2 * live as u32)).max(1);
        for _ in 0..extra {
            sequence.extend_from_slice(&visits[..live]);
            for v in visits.iter_mut().take(live) {
                *v += 1;
            }
        }
        live = (live / 2).max(2);
    }
    sequence.truncate(simulations as usize);
    sequence
}

/// Standard Gumbel(0, 1) sample.
pub fn gumbel_sample<R: Rng + RngExt>(rng: &mut R) -> f32 {
    let u: f64 = rng.random_range(1e-12..1.0);
    (-(-u.ln()).ln()) as f32
}

impl MCTSTree {
    /// Q of a child for the side to move at its parent: exact for proven children,
    /// the search average otherwise, `None` when the child has no information.
    fn child_q(&self, child: NodeId) -> Option<f32> {
        let node = self.get_node(child);
        if node.proven != 0 {
            // `proven` is from the child's own side; the parent sees the opposite.
            return Some(-(node.proven as f32));
        }
        if node.visits > 0.0 {
            Some(node.wins / node.visits)
        } else {
            None
        }
    }

    /// Visits of a child, with virtual losses in flight when `in_flight` is set.
    fn child_visits(&self, child: NodeId, in_flight: bool) -> f32 {
        let node = self.get_node(child);
        if in_flight {
            node.visits + node.virtual_loss
        } else {
            node.visits
        }
    }

    /// sigma(completed Q) for every child of `node_id`, in children order.
    pub(crate) fn sigma_q(&self, node_id: NodeId, cfg: &GumbelConfig) -> Vec<f32> {
        let node = self.get_node(node_id);
        let children = &node.children;
        let raw = node.raw_value;

        let mut visits_total = 0.0f32;
        let mut prior_visited = 0.0f32;
        let mut weighted = 0.0f32;
        let mut max_visits = 0.0f32;
        for &c in children {
            let n = self.child_visits(c, false);
            max_visits = max_visits.max(n);
            if let Some(q) = self.child_q(c) {
                let child = self.get_node(c);
                let n_eff = if child.proven != 0 { n.max(1.0) } else { n };
                visits_total += n_eff;
                prior_visited += child.prior;
                weighted += child.prior * q;
            }
        }
        let mixed = if prior_visited > 0.0 {
            (raw + visits_total * weighted / prior_visited) / (visits_total + 1.0)
        } else {
            raw
        };

        let mut completed: Vec<f32> = children
            .iter()
            .map(|&c| self.child_q(c).unwrap_or(mixed))
            .collect();
        let lo = completed.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = completed.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = (cfg.maxvisit_init + max_visits) * cfg.value_scale / (hi - lo).max(1e-8);
        for q in &mut completed {
            *q = (*q - lo) * scale;
        }
        completed
    }

    /// Improved policy `softmax(logit + sigma)` over the children of `node_id`.
    pub(crate) fn improved_policy(&self, node_id: NodeId, cfg: &GumbelConfig) -> Vec<f32> {
        let node = self.get_node(node_id);
        let sigma = self.sigma_q(node_id, cfg);
        let scores: Vec<f32> = node
            .children
            .iter()
            .zip(&sigma)
            .map(|(&c, s)| self.get_node(c).logit + s)
            .collect();
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let total: f32 = exps.iter().sum();
        exps.iter().map(|x| x / total).collect()
    }

    /// Interior selection: the child whose visit share lags its improved-policy
    /// probability the most (mctx `gumbel_muzero_interior_action_selection`).
    pub(crate) fn gumbel_select_interior(&self, node_id: NodeId, cfg: &GumbelConfig) -> NodeId {
        let node = self.get_node(node_id);
        let pi = self.improved_policy(node_id, cfg);
        let visits: Vec<f32> = node
            .children
            .iter()
            .map(|&c| self.child_visits(c, true))
            .collect();
        let denom = 1.0 + visits.iter().sum::<f32>();
        let mut best = (f32::NEG_INFINITY, node.children[0]);
        for (i, &c) in node.children.iter().enumerate() {
            let score = pi[i] - visits[i] / denom;
            if score > best.0 {
                best = (score, c);
            }
        }
        best.1
    }

    /// Root selection by sequential halving: among the children at the scheduled
    /// visit count, the best `gumbel + logit + sigma`.
    pub(crate) fn gumbel_select_root(&self, cfg: &GumbelConfig) -> NodeId {
        let root_id = self.get_root_id();
        let root = self.get_node(root_id);
        let visits: Vec<f32> = root
            .children
            .iter()
            .map(|&c| self.child_visits(c, true))
            .collect();
        let done = visits.iter().sum::<f32>().round() as usize;
        let wanted = match self.gumbel_schedule.get(done) {
            Some(&v) => v as f32,
            None => visits.iter().copied().fold(0.0, f32::max),
        };
        let sigma = self.sigma_q(root_id, cfg);
        let mut best = (f32::NEG_INFINITY, None);
        for (i, &c) in root.children.iter().enumerate() {
            if (visits[i] - wanted).abs() > 0.5 {
                continue;
            }
            let score = self.gumbel_noise.get(i).copied().unwrap_or(0.0)
                + self.get_node(c).logit
                + sigma[i];
            if score > best.0 {
                best = (score, Some(c));
            }
        }
        // Virtual losses can leave no child at the scheduled count: fall back to the
        // interior rule.
        best.1
            .unwrap_or_else(|| self.gumbel_select_interior(root_id, cfg))
    }

    /// The move to play: among the most visited root children, the best
    /// `gumbel + logit + sigma`. Proven wins are taken first, proven losses avoided
    /// while anything else exists.
    pub(crate) fn gumbel_best_child(&self, cfg: &GumbelConfig) -> Option<NodeId> {
        let root_id = self.get_root_id();
        let root = self.get_node(root_id);
        if root.children.is_empty() {
            return None;
        }
        if let Some(&win) = root
            .children
            .iter()
            .filter(|&&c| self.get_node(c).proven == -1)
            .min_by_key(|&&c| self.get_node(c).proven_dist)
        {
            return Some(win);
        }
        let pool: Vec<NodeId> = {
            let non_losing: Vec<NodeId> = root
                .children
                .iter()
                .copied()
                .filter(|&c| self.get_node(c).proven != 1)
                .collect();
            if non_losing.is_empty() {
                root.children.clone()
            } else {
                non_losing
            }
        };
        let max_visits = pool
            .iter()
            .map(|&c| self.get_node(c).visits)
            .fold(0.0, f32::max);
        let sigma = self.sigma_q(root_id, cfg);
        let index_of = |c: NodeId| root.children.iter().position(|&x| x == c).unwrap();
        pool.iter()
            .copied()
            .filter(|&c| (self.get_node(c).visits - max_visits).abs() < 0.5)
            .max_by(|&a, &b| {
                let score = |c: NodeId| {
                    let i = index_of(c);
                    self.gumbel_noise.get(i).copied().unwrap_or(0.0)
                        + self.get_node(c).logit
                        + sigma[i]
                };
                score(a).total_cmp(&score(b))
            })
    }

    /// Prepares the root for sequential halving over `simulations` root visits.
    pub(crate) fn init_gumbel_root<R: Rng + RngExt>(
        &mut self,
        cfg: &GumbelConfig,
        simulations: u32,
        rng: &mut R,
    ) {
        let count = self.get_root().children.len();
        self.gumbel_noise = (0..count)
            .map(|_| if cfg.noise { gumbel_sample(rng) } else { 0.0 })
            .collect();
        let considered = cfg.considered.min(count).max(1);
        self.gumbel_schedule = halving_schedule(considered, simulations.max(1));
        self.gumbel_cfg = Some(*cfg);
    }

    /// Training target over the root moves: the improved policy, `None` when the
    /// last search was not a Gumbel search.
    pub fn gumbel_policy_target(&self) -> Option<Vec<(crate::mv::Move, f32)>> {
        let cfg = self.gumbel_cfg?;
        let root_id = self.get_root_id();
        let root = self.get_node(root_id);
        if root.children.is_empty() {
            return Some(vec![]);
        }
        let pi = self.improved_policy(root_id, &cfg);
        Some(
            root.children
                .iter()
                .zip(pi)
                .map(|(&c, p)| (self.get_node(c).mv.unwrap(), p))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The schedule for 16 considered actions and 159 simulations, as mctx computes
    // it: four rounds of halving (16 → 8 → 4 → 2), each round spending about a
    // quarter of the budget.
    #[test]
    fn halving_schedule_matches_mctx() {
        let s = halving_schedule(16, 159);
        assert_eq!(s.len(), 159);
        // Round 1: 159 / (4 * 16) = 2 passes over 16 actions → visits 0 and 1.
        assert_eq!(&s[..16], &[0; 16]);
        assert_eq!(&s[16..32], &[1; 16]);
        // Round 2: 159 / (4 * 8) = 4 passes over 8 actions → visits 2..=5.
        for pass in 0..4u32 {
            let from = 32 + 8 * pass as usize;
            assert_eq!(&s[from..from + 8], &[2 + pass; 8]);
        }
        // Round 3: 159 / 16 = 9 passes over 4 → 6..=14; round 4 fills the rest on 2.
        assert_eq!(s[64], 6);
        assert_eq!(s[64 + 35], 14);
        assert_eq!(s[100], 15);
        assert_eq!(*s.last().unwrap(), 15 + (159 - 100 - 1) as u32 / 2);
        // Every count is reached in order, never skipped.
        let mut seen = std::collections::HashMap::new();
        for &v in &s {
            *seen.entry(v).or_insert(0) += 1;
        }
        for v in 0..=*s.iter().max().unwrap() {
            assert!(seen.contains_key(&v), "visit count {v} never scheduled");
        }
    }

    #[test]
    fn halving_schedule_degenerate_cases() {
        assert_eq!(halving_schedule(1, 5), vec![0, 1, 2, 3, 4]);
        assert_eq!(halving_schedule(2, 3), vec![0, 0, 1]);
        assert_eq!(halving_schedule(16, 0), Vec::<u32>::new());
    }

    #[test]
    fn gumbel_samples_have_the_right_mean() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let n = 20_000;
        let mean: f64 = (0..n).map(|_| gumbel_sample(&mut rng) as f64).sum::<f64>() / n as f64;
        // Euler–Mascheroni constant, loosely.
        assert!((mean - 0.5772).abs() < 0.03, "mean {mean}");
    }

    use rand::SeedableRng;
}

#[cfg(test)]
mod search_tests {
    use super::*;
    use crate::board::Board;
    use crate::board::rules::RulesEnum;
    use crate::mcts::{MCTSConfig, mcts_search};
    use crate::search::nn::NeuralNet;
    use crate::search_data::SearchData;

    fn start_board() -> Board {
        let mut b = Board::new();
        b.set_rules(RulesEnum::Copenhagen11x11);
        b.setup_initial_position().unwrap();
        b
    }

    fn gumbel_config(noise: bool) -> MCTSConfig {
        let mut c = MCTSConfig::default_train();
        c.dirichlet_alpha = 0.0;
        c.dirichlet_epsilon = 0.0;
        c.forced_playouts_k = 0.0;
        c.gumbel = Some(GumbelConfig {
            considered: 16,
            noise,
            ..GumbelConfig::default()
        });
        c
    }

    // 160 simulations from the start: the root budget lands on at most 16 children,
    // visits add up to the budget, and the policy target is a distribution over
    // every legal move.
    #[test]
    fn gumbel_search_spends_budget_on_considered_actions() {
        let mut nn = NeuralNet::new("copenhagen11x11.onnx", 11, 8).unwrap();
        let mut board = start_board();
        let mut tree = MCTSTree::new();
        let mut sd = SearchData::new();
        let cfg = gumbel_config(true);
        let mv = mcts_search(
            &mut board,
            &mut tree,
            &mut nn,
            &mut sd,
            None,
            Some(160),
            &cfg,
            None,
        );
        assert!(mv.is_some());

        let root = tree.get_root();
        let visited: Vec<f32> = root
            .children()
            .iter()
            .map(|&c| tree.get_node(c).visits())
            .filter(|&v| v > 0.0)
            .collect();
        let total: f32 = visited.iter().sum();
        assert_eq!(total.round() as u32, 160);
        assert!(visited.len() <= 16, "{} children visited", visited.len());
        assert!(visited.len() >= 2);

        let target = tree.gumbel_policy_target().unwrap();
        assert_eq!(target.len(), root.children().len());
        let sum: f32 = target.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-3, "policy sums to {sum}");
        assert!(target.iter().all(|(_, p)| *p >= 0.0));
    }

    // Without root noise the search is a deterministic function of the position.
    #[test]
    fn gumbel_without_noise_is_deterministic() {
        let mut nn = NeuralNet::new("copenhagen11x11.onnx", 11, 8).unwrap();
        let cfg = gumbel_config(false);
        let mut moves = vec![];
        for _ in 0..2 {
            let mut board = start_board();
            let mut tree = MCTSTree::new();
            let mut sd = SearchData::new();
            moves.push(
                mcts_search(
                    &mut board,
                    &mut tree,
                    &mut nn,
                    &mut sd,
                    None,
                    Some(40),
                    &cfg,
                    None,
                )
                .unwrap(),
            );
        }
        assert_eq!(moves[0], moves[1]);
    }

    // sigma(completed Q): visited children keep their own Q, unvisited ones take the
    // mixed value; the transform is affine in Q so order is preserved.
    #[test]
    fn sigma_q_completes_unvisited_children_with_the_mixed_value() {
        let mut nn = NeuralNet::new("copenhagen11x11.onnx", 11, 8).unwrap();
        let mut board = start_board();
        let mut tree = MCTSTree::new();
        let mut sd = SearchData::new();
        let cfg = gumbel_config(false);
        mcts_search(
            &mut board,
            &mut tree,
            &mut nn,
            &mut sd,
            None,
            Some(24),
            &cfg,
            None,
        );
        let g = cfg.gumbel.unwrap();
        let root_id = tree.get_root_id();
        let sigma = tree.sigma_q(root_id, &g);
        let root = tree.get_root();
        assert_eq!(sigma.len(), root.children().len());
        // All unvisited children share one sigma (the mixed value).
        let unvisited: Vec<f32> = root
            .children()
            .iter()
            .zip(&sigma)
            .filter(|(c, _)| tree.get_node(**c).visits() == 0.0 && tree.get_node(**c).proven == 0)
            .map(|(_, s)| *s)
            .collect();
        assert!(unvisited.len() > 1);
        assert!(unvisited.iter().all(|&s| (s - unvisited[0]).abs() < 1e-5));
        // Among visited children, sigma orders like Q.
        let mut visited: Vec<(f32, f32)> = root
            .children()
            .iter()
            .zip(&sigma)
            .filter(|(c, _)| tree.get_node(**c).visits() > 0.0)
            .map(|(c, s)| (tree.get_node(*c).wins() / tree.get_node(*c).visits(), *s))
            .collect();
        visited.sort_by(|a, b| a.0.total_cmp(&b.0));
        for w in visited.windows(2) {
            assert!(w[0].1 <= w[1].1 + 1e-6);
        }
        assert!(sigma.iter().all(|&s| s >= -1e-6));
    }
}
