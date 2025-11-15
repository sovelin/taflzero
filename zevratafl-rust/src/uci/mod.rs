use js_sys::global;
use wasm_bindgen::prelude::*;
use web_sys::{window, CustomEvent, CustomEventInit};
use crate::constants::INITIAL_FEN;
use crate::Engine;
use crate::mv::{create_move_from_algebraic, Move};
use crate::nnue::{load_fc1_from_raw, load_fc2_from_raw};

#[wasm_bindgen]
pub struct WasmClient {
    event_name: String,
    engine: Engine,
}

fn broadcast(event_name: &str, msg: &str) {
    let global = global(); // <-- это работает и в window, и в worker

    let init = CustomEventInit::new();
    init.set_detail(&JsValue::from_str(msg));

    let event = CustomEvent::new_with_event_init_dict(event_name, &init).unwrap();

    let _ = global
        .dyn_ref::<web_sys::EventTarget>()
        .unwrap()
        .dispatch_event(&event);
}
#[wasm_bindgen]
impl WasmClient {
    #[wasm_bindgen(constructor)]
    pub fn new(event_name: String, tt_size: usize) -> Self {
        let w1 = load_fc1_from_raw();
        let w2 = load_fc2_from_raw();

        Self { event_name, engine: Engine::new(tt_size, &w1, &w2) }
    }

    fn set_moves(&mut self, fen: &str, moves_str: &[&str]) {
        let moves: Vec<_> = moves_str
            .iter()
            .map(|mv_str| create_move_from_algebraic(mv_str))
            .collect();

        let mut legal_moves = Vec::new();

        for mv_res in &moves {
            match mv_res {
                Ok(mv) => legal_moves.push(*mv),
                Err(_) => {
                    self.broadcast("invalid move in moves list");
                    return;
                }
            }
        }

        if let Err(_) = self.engine.set_position_and_moves(fen, legal_moves) {
            self.broadcast("invalid fen string");
            return;
        }
    }

    fn handle_position(&mut self, args: &[&str]) {
        match args[0] {
            "startpos" => {
                if args.len() < 2 {
                    self.set_moves(INITIAL_FEN, &[]);
                    return;
                }

                if args[1] == "moves" {
                    self.set_moves(INITIAL_FEN, &args[2..]);
                    return;
                }
            }
            "fen" => {
                let fen = args[1];

                if args[2] != "moves" {
                    self.broadcast("only 'position fen <fen> moves' without moves is supported");
                    return;
                }

                self.set_moves(fen, &args[3..]);
            },
            _ => {
                self.broadcast("unknown position command");
            }
        }
    }

    fn handle_go(&mut self, args: &[&str]) {
        match args[0] {
            "movetime" => {
                if args.len() < 2 {
                    self.broadcast("invalid movetime command");
                    return;
                }
                let movetime = args[1].parse::<u64>().unwrap_or(0);

                if movetime == 0 {
                    self.broadcast("invalid movetime value");
                    return;
                }

                self.engine.make_search(movetime, Some(&|iteration| {
                    let msg = format!(
                        "info depth {} score {} nodes {} time {} speed {} bestmove {:?}",
                        iteration.depth,
                        iteration.score,
                        iteration.nodes,
                        iteration.time,
                        iteration.speed,
                        iteration.mv,
                    );

                    broadcast(&self.event_name, &msg);
                }));

                if let Some(mv) = self.engine.best_move() {
                    self.broadcast(&format!("bestmove {:?}", mv));
                }
            },
            _ => {
                self.broadcast("unknown go command");
            }
        }
    }

    #[wasm_bindgen]
    pub fn run(&mut self, cmd: &str) {
        let tokens = cmd.split_whitespace().collect::<Vec<&str>>();

        if tokens.is_empty() {
            self.broadcast("unknown command");
            return;
        }

        let cmd = tokens[0];

        // print cmd
        let str = format!("Command received: {}", cmd);
        self.broadcast(&str);
        let str2 = format!("Full command: {}", cmd);
        self.broadcast(&str2);

        let response = match cmd {
            "isready" => {
                self.broadcast("readyok");
            },
            "uci" => {
                self.broadcast("id name ZevraTafl\nid author Oleg Smirnov\nuciok");
            },
            "position" => {
                self.handle_position(&tokens[1..]);
                return;
            },
            "board" => {
                self.broadcast(&format!("{:?}", self.engine.board()));
            },
            "go" => {
                self.handle_go(&tokens[1..]);
                return;
            },
            _ => {
                self.broadcast("unknown command");
            },
        };
    }

    fn broadcast(&self, msg: &str) {
        broadcast(&self.event_name, msg);
    }
}
