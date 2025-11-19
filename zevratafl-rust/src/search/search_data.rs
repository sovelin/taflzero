use rand::rngs::StdRng;
use rand::SeedableRng;
use crate::movegen::MAX_MOVES;
use crate::moves::movegen::MoveGen;
use crate::moves::mv::Move;
use crate::moves::undo::UndoMove;
use crate::search::constants::MAX_PLY;
use crate::search::history::History;
use crate::search::killer::Killer;
use crate::timer::Timer;

pub struct SearchData {
    pub nodes_searched: u64,
    pub best_move: Option<Move>,
    pub move_gens: Vec<MoveGen>,
    pub undos: Vec<UndoMove>,
    pub history: History,
    pub killers: Killer,
    pub timer: Timer,
    pub time_limit: u64,
    cached_exceed: bool,
    time_exceeded_checks: u32,
    pub temperatures: Vec<Vec<i32>>,
    pub temperature: usize,
    pub random_generator: StdRng,
}

impl SearchData {
    pub fn new() -> Self {
        let mut move_gens = Vec::with_capacity(MAX_PLY);
        let mut undos = Vec::with_capacity(MAX_PLY);

        for _ in 0..MAX_PLY {
            move_gens.push(MoveGen::new());
            undos.push(UndoMove::new());
        }

        let mut temperatures = Vec::with_capacity(MAX_PLY);
        for _ in 0..MAX_PLY {
            temperatures.push(vec![0; MAX_MOVES]);
        }

        Self {
            nodes_searched: 0,
            best_move: None,
            move_gens,
            undos,
            timer: Timer::new(),
            time_limit: 0,
            history: History::new(),
            killers: Killer::new(),
            cached_exceed: false,
            time_exceeded_checks: 0,
            temperatures,
            //temperature: 15,
            temperature: 0,
            random_generator: StdRng::seed_from_u64(123456),
        }
    }

    pub fn time_exceeded(&mut self) -> bool {
        self.timer.elapsed_ms() >= self.time_limit
    }

    pub fn time_exceeded_quick(&mut self) -> bool {
        self.time_exceeded_checks += 1;
        if self.time_exceeded_checks < 10000 {
            return self.cached_exceed
        }

        self.time_exceeded_checks = 0;
        self.cached_exceed = self.timer.elapsed_ms() >= self.time_limit;
        self.cached_exceed
    }

    pub fn start_timer(&mut self, time_limit_ms: u64) {
        self.timer.start();
        self.time_limit = time_limit_ms;
        self.time_exceeded_checks = 0;
        self.cached_exceed = false;
    }
}