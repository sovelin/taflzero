pub mod engine_client;
pub mod constants;

use crate::mv::create_move_from_algebraic;
use crate::nnue::{load_fc1_from_raw, load_fc2_from_raw};
use crate::search::search_root::SearchIterationResponse;
use crate::Engine;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UciRunState {
    Continue,
    Quit,
}

pub trait UciOutput {
    fn send(&self, message: &str);
}

pub struct UciController<O: UciOutput> {
    engine: Engine,
    output: O,
}

impl<O: UciOutput> UciController<O> {
    pub fn new(tt_size_mb: usize, output: O) -> Self {
        let w1 = load_fc1_from_raw();
        let w2 = load_fc2_from_raw();

        Self {
            engine: Engine::new(tt_size_mb, &w1, &w2),
            output,
        }
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    fn send(&self, message: &str) {
        self.output.send(message);
    }

    pub fn run(&mut self, cmd: &str) -> UciRunState {
        let tokens = cmd.split_whitespace().collect::<Vec<&str>>();

        if tokens.is_empty() {
            self.send("unknown command");
            return UciRunState::Continue;
        }

        let keyword = tokens[0];
        self.send(&format!("Command received: {}", keyword));
        self.send(&format!("Full command: {}", cmd));

        match keyword {
            "quit" => {
                self.send("bye");
                UciRunState::Quit
            }
            "isready" => {
                self.send("readyok");
                UciRunState::Continue
            }
            "uci" => {
                self.send("id name ZevraTafl\nid author Oleg Smirnov\nuciok");
                UciRunState::Continue
            }
            "position" => {
                self.handle_position(&tokens[1..]);
                UciRunState::Continue
            }
            "board" => {
                self.send(&format!("{:?}", self.engine.board()));
                self.send(&format!("Eval: {}", self.engine.board().get_eval()));
                UciRunState::Continue
            }
            "go" => {
                self.handle_go(&tokens[1..]);
                UciRunState::Continue
            }
            _ => {
                self.send("unknown command");
                UciRunState::Continue
            }
        }
    }

    fn handle_position(&mut self, args: &[&str]) {
        if args.is_empty() {
            self.send("position command missing arguments");
            return;
        }

        match args[0] {
            "startpos" => {
                if args.len() == 1 {
                    self.set_moves(INITIAL_FEN, &[]);
                    self.send("position set to startpos");
                    return;
                }

                if args[1] == "moves" {
                    self.set_moves(INITIAL_FEN, &args[2..]);
                    self.send(&format!("position set to startpos ({} moves)", args.len() - 2));
                    return;
                }

                self.send("unsupported startpos format");
            }
            "fen" => {
                let fen = format!("{} {}", args[1], args[2]);

                if args.len() < 4 {
                    self.set_moves(&fen, &vec![]);
                    return;
                }


                if args[3] != "moves" {
                    self.send("only 'position fen <fen> moves' is supported");
                    return;
                }

                self.set_moves(&fen, &args[4..]);
                self.send(&format!("position set to fen '{}' ({} moves)", fen, args.len() - 4));
            }
            _ => {
                self.send("unknown position command");
            }
        }
    }

    fn set_moves(&mut self, fen: &str, moves_str: &[&str]) {
        let mut legal_moves = Vec::with_capacity(moves_str.len());

        for mv_str in moves_str {
            match create_move_from_algebraic(mv_str) {
                Ok(mv) => legal_moves.push(mv),
                Err(err) => {
                    self.send(&format!("invalid move '{}': {}", mv_str, err));
                    return;
                }
            }
        }

        self.engine.set_position_and_moves(fen, legal_moves);
    }

    fn handle_go(&mut self, args: &[&str]) {
        if args.is_empty() {
            self.send("go command missing arguments");
            return;
        }

        match args[0] {
            "movetime" => self.handle_go_movetime(&args[1..]),
            _ => self.send("unknown go subcommand"),
        }
    }

    fn handle_go_movetime(&mut self, args: &[&str]) {
        if args.is_empty() {
            self.send("movetime value missing");
            return;
        }

        let movetime = args[0].parse::<u64>().unwrap_or(0);

        if movetime == 0 {
            self.send("invalid movetime value");
            return;
        }

        let output = &self.output;
        self.engine.make_search(movetime, MAX_PLY as u32, Some(&|iteration: SearchIterationResponse| {
            let msg = format!(
                "info depth {} score {} nodes {} time {} speed {} bestmove {:?}",
                iteration.depth,
                iteration.score,
                iteration.nodes,
                iteration.time,
                iteration.speed,
                iteration.mv,
            );
            output.send(&msg);
        }));

        if let Some(mv) = self.engine.best_move() {
            self.send(&format!("bestmove {:?}", mv));
        } else {
            self.send("bestmove (none)");
        }
    }
}

#[derive(Default, Clone, Copy)]
pub struct ConsoleBridge;

impl UciOutput for ConsoleBridge {
    fn send(&self, message: &str) {
        println!("{}", message);
    }
}

pub struct ConsoleClient {
    controller: UciController<ConsoleBridge>,
}

impl ConsoleClient {
    pub fn new(tt_size_mb: usize) -> Self {
        Self {
            controller: UciController::new(tt_size_mb, ConsoleBridge),
        }
    }

    pub fn run_line(&mut self, line: &str) -> UciRunState {
        self.controller.run(line)
    }

    pub fn engine(&self) -> &Engine {
        self.controller.engine()
    }
}

#[cfg(target_arch = "wasm32")]
use js_sys::global;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{prelude::*, JsCast};
#[cfg(target_arch = "wasm32")]
use web_sys::{CustomEvent, CustomEventInit};
use crate::board::constants::INITIAL_FEN;
use crate::search::constants::MAX_PLY;

#[cfg(target_arch = "wasm32")]
fn broadcast(event_name: &str, msg: &str) {
    let global = global();

    let init = CustomEventInit::new();
    init.set_detail(&JsValue::from_str(msg));

    let event = CustomEvent::new_with_event_init_dict(event_name, &init).unwrap();

    let _ = global
        .dyn_ref::<web_sys::EventTarget>()
        .unwrap()
        .dispatch_event(&event);
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone)]
struct WasmBridge {
    event_name: String,
}

#[cfg(target_arch = "wasm32")]
impl WasmBridge {
    fn new(event_name: String) -> Self {
        Self { event_name }
    }
}

#[cfg(target_arch = "wasm32")]
impl UciOutput for WasmBridge {
    fn send(&self, message: &str) {
        broadcast(&self.event_name, message);
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmClient {
    controller: UciController<WasmBridge>,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmClient {
    #[wasm_bindgen(constructor)]
    pub fn new(event_name: String, tt_size: usize) -> Self {
        let bridge = WasmBridge::new(event_name);
        Self {
            controller: UciController::new(tt_size, bridge),
        }
    }

    pub fn print_board(&self) {
        self.controller.engine().print_board();
    }

    pub fn run(&mut self, cmd: &str) {
        self.controller.run(cmd);
    }
}
