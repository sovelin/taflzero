pub mod engine_client;
pub mod constants;

use crate::mv::create_move_from_algebraic;
use crate::search::search_root::SearchIterationResponse;
use crate::Engine;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UciRunState {
    Continue,
    Quit,
}

pub trait UciOutput: Clone + Send + 'static {
    fn send(&self, message: &str);
}

#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

#[cfg(not(target_arch = "wasm32"))]
struct SearchThread {
    handle: std::thread::JoinHandle<Engine>,
    stop_flag: Arc<AtomicBool>,
    bestmove_sent: Arc<AtomicBool>,
}

pub struct UciController<O: UciOutput> {
    engine: Option<Engine>,
    output: O,
    #[cfg(not(target_arch = "wasm32"))]
    search_thread: Option<SearchThread>,
}

fn format_info_message(iteration: SearchIterationResponse) -> String {
    let pv_str = iteration.pv().iter().map(|m| format!("{:?}", m)).collect::<Vec<_>>().join(" ");

    let multipv_str = if let Some(mpv) = iteration.multi_pv {
        format!(" multipv {}", mpv)
    } else {
        String::new()
    };

    format!(
        "info{} score cp {} winrate {:.1}% nodes {} time {} speed {} pv {}",
        multipv_str,
        iteration.score,
        iteration.winrate * 100.0,
        iteration.nodes,
        iteration.time,
        iteration.speed,
        pv_str,
    )
}

impl<O: UciOutput> UciController<O> {
    pub fn new(output: O, net_path: String) -> Self {
        Self {
            engine: Some(Engine::new(net_path)),
            output,
            #[cfg(not(target_arch = "wasm32"))]
            search_thread: None,
        }
    }

    pub fn engine(&self) -> &Engine {
        self.engine.as_ref().expect("engine is busy in search thread")
    }

    pub fn engine_mut(&mut self) -> &mut Engine {
        self.engine.as_mut().expect("engine is busy in search thread")
    }

    fn send(&self, message: &str) {
        self.output.send(message);
    }

    // Stops any running search, joins the thread, and restores the engine.
    // Sends bestmove if send_bestmove is true and it hasn't been sent already.
    #[cfg(not(target_arch = "wasm32"))]
    fn collect_search(&mut self, send_bestmove: bool) {
        let thread = match self.search_thread.take() {
            Some(t) => t,
            None => return,
        };

        thread.stop_flag.store(true, Ordering::Relaxed);
        let mut engine = thread.handle.join().expect("search thread panicked");
        engine.clear_stop_flag();

        if send_bestmove && !thread.bestmove_sent.swap(true, Ordering::SeqCst) {
            if let Some(mv) = engine.best_move() {
                self.output.send(&format!("bestmove {:?}", mv));
            } else {
                self.output.send("bestmove (none)");
            }
        }

        self.engine = Some(engine);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn spawn_search<F>(&mut self, search_fn: F)
    where
        F: FnOnce(&mut Engine) + Send + 'static,
    {
        self.collect_search(false);

        let stop_flag = Arc::new(AtomicBool::new(false));
        let bestmove_sent = Arc::new(AtomicBool::new(false));

        let mut engine = self.engine.take().expect("engine unavailable");
        engine.set_stop_flag(Arc::clone(&stop_flag));

        let output = self.output.clone();
        let bestmove_sent_clone = Arc::clone(&bestmove_sent);

        let handle = std::thread::spawn(move || {
            search_fn(&mut engine);

            if !bestmove_sent_clone.swap(true, Ordering::SeqCst) {
                if let Some(mv) = engine.best_move() {
                    output.send(&format!("bestmove {:?}", mv));
                } else {
                    output.send("bestmove (none)");
                }
            }

            engine
        });

        self.search_thread = Some(SearchThread { handle, stop_flag, bestmove_sent });
    }

    pub fn run(&mut self, cmd: &str) -> UciRunState {
        let tokens = cmd.split_whitespace().collect::<Vec<&str>>();

        if tokens.is_empty() {
            self.send("unknown command");
            return UciRunState::Continue;
        }

        let keyword = tokens[0];

        match keyword {
            "quit" => {
                #[cfg(not(target_arch = "wasm32"))]
                self.collect_search(false);
                self.send("bye");
                UciRunState::Quit
            }
            "stop" => {
                #[cfg(not(target_arch = "wasm32"))]
                self.collect_search(true);
                UciRunState::Continue
            }
            "isready" => {
                self.send("readyok");
                UciRunState::Continue
            }
            "uci" => {
                self.send(&format!(
                    "id name {}\nid author Oleg Smirnov\nuciok",
                    env!("CARGO_PKG_NAME")
                ));
                self.send("option name NNFile type string default ./default_nn.onnx");
                self.send("option name MultiPV type spin default 1 min 1 max 1000");
                self.send("option name Variant type combo default copenhagen11x11 var copenhagen11x11 var historical11x11");
                UciRunState::Continue
            }
            "setoption" => {
                if tokens.len() >= 5 && tokens[1] == "name" && tokens[2] == "NNFile" && tokens[3] == "value" {
                    let path = tokens[4];
                    #[cfg(not(target_arch = "wasm32"))]
                    self.collect_search(false);
                    self.engine_mut().set_nn(path.to_string());
                    self.send(&format!("NN file set to '{}'", path));
                } else if tokens.len() >= 5 && tokens[1] == "name" && tokens[2] == "MultiPV" && tokens[3] == "value" {
                    let multipv = tokens[4].parse::<usize>().unwrap_or(1);

                    if multipv == 0 || multipv > 1000 {
                        self.send("invalid MultiPV value");
                        return UciRunState::Continue;
                    }

                    self.engine_mut().set_multi_pv(multipv);
                    self.send(&format!("MultiPV set to {}", multipv));
                } else if tokens.len() >= 5 && tokens[1] == "name" && tokens[2] == "Variant" && tokens[3] == "value" {
                    let variant = tokens[4];
                    match variant {
                        "copenhagen11x11" => {
                            self.engine_mut().set_variant(RulesEnum::Copenhagen11x11);
                            self.send("variant set to copenhagen11x11");
                        }
                        "historical11x11" => {
                            self.engine_mut().set_variant(RulesEnum::Historical11x11);
                            self.send("variant set to historical11x11");
                        }
                        _ => self.send("unknown variant"),
                    }
                }

                else {
                    self.send("unsupported setoption format");
                }
                UciRunState::Continue
            }
            "position" => {
                #[cfg(not(target_arch = "wasm32"))]
                self.collect_search(false);
                self.handle_position(&tokens[1..]);
                UciRunState::Continue
            }
            "board" => {
                self.send(&format!("{:?}", self.engine().board()));
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
                    self.set_initial_moves(&[]);
                    self.send("position set to startpos");
                    return;
                }

                if args[1] == "moves" {
                    self.set_initial_moves(&args[2..]);
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

        self.engine_mut().set_position_and_moves(fen, legal_moves);
    }

    fn set_initial_moves(&mut self, moves_str: &[&str]) {
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

        self.engine_mut().set_initial_position_and_moves(legal_moves);
    }

    fn handle_go(&mut self, args: &[&str]) {
        if args.is_empty() {
            self.send("go command missing arguments");
            return;
        }

        match args[0] {
            "movetime" => self.handle_go_movetime(&args[1..]),
            "nodes" => self.handle_go_nodes(&args[1..]),
            "infinite" => self.handle_go_infinite(),
            _ => self.send("unknown go subcommand"),
        }
    }

    fn handle_go_nodes(&mut self, args: &[&str]) {
        if args.is_empty() {
            self.send("nodes value missing");
            return;
        }

        let nodes = args[0].parse::<u64>().unwrap_or(0);

        if nodes == 0 {
            self.send("invalid nodes value");
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let output = self.output.clone();
            self.spawn_search(move |engine| {
                engine.make_search_nodes(nodes, Some(&|iteration: SearchIterationResponse| {
                    output.send(&format_info_message(iteration));
                }));
            });
        }

        #[cfg(target_arch = "wasm32")]
        {
            let output = self.output.clone();
            self.engine_mut().make_search_nodes(nodes, Some(&|iteration: SearchIterationResponse| {
                output.send(&format_info_message(iteration));
            }));
            if let Some(mv) = self.engine().best_move() {
                self.send(&format!("bestmove {:?}", mv));
            } else {
                self.send("bestmove (none)");
            }
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

        #[cfg(not(target_arch = "wasm32"))]
        {
            let output = self.output.clone();
            self.spawn_search(move |engine| {
                engine.make_search(movetime, MAX_PLY as u32, Some(&|iteration: SearchIterationResponse| {
                    output.send(&format_info_message(iteration));
                }));
            });
        }

        #[cfg(target_arch = "wasm32")]
        {
            let output = self.output.clone();
            self.engine_mut().make_search(movetime, MAX_PLY as u32, Some(&|iteration: SearchIterationResponse| {
                output.send(&format_info_message(iteration));
            }));
            if let Some(mv) = self.engine().best_move() {
                self.send(&format!("bestmove {:?}", mv));
            } else {
                self.send("bestmove (none)");
            }
        }
    }

    fn handle_go_infinite(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let output = self.output.clone();
            self.spawn_search(move |engine| {
                engine.make_search_infinite(Some(&|iteration: SearchIterationResponse| {
                    output.send(&format_info_message(iteration));
                }));
            });
        }

        #[cfg(target_arch = "wasm32")]
        {
            let output = self.output.clone();
            self.engine_mut().make_search_infinite(Some(&|iteration: SearchIterationResponse| {
                output.send(&format_info_message(iteration));
            }));
            if let Some(mv) = self.engine().best_move() {
                self.send(&format!("bestmove {:?}", mv));
            } else {
                self.send("bestmove (none)");
            }
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
    pub fn new(net_path: String) -> Self {
        Self {
            controller: UciController::new(ConsoleBridge, net_path)
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
use crate::rules::RulesEnum;
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
    pub fn new(event_name: String) -> Self {
        let bridge = WasmBridge::new(event_name);
        Self {
            controller: UciController::new(bridge, "".to_string()),
        }
    }

    pub fn print_board(&self) {
        self.controller.engine().print_board();
    }

    pub fn run(&mut self, cmd: &str) {
        self.controller.run(cmd);
    }

    /// Register a SharedArrayBuffer-backed Int32Array as the stop signal.
    /// The main thread can stop an ongoing `go infinite` by calling:
    ///   `Atomics.store(buffer, 0, 1)`
    /// Reset before each new search with `Atomics.store(buffer, 0, 0)`.
    pub fn set_stop_buffer(&mut self, buffer: js_sys::Int32Array) {
        crate::search::search_data::set_wasm_stop_buffer(buffer);
    }
}
