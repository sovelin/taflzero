pub struct Rules {
    pub has_corners_win: bool,
    pub has_edge_win: bool,
    pub has_fort_win: bool,
    pub has_shield_walls: bool,
    pub initial_fen: String,
    pub is_king_strong: bool,
}

pub enum RulesEnum {
    Copenhagen11x11,
    Historical11x11,
}

impl RulesEnum {
    pub fn rules(&self) -> Rules {
        match self {
            RulesEnum::Copenhagen11x11 => Rules::create_copenhagen_rules(),
            RulesEnum::Historical11x11 => Rules::create_historical_rules(),
        }
    }
}


impl Rules {
    pub fn create_copenhagen_rules() -> Rules {
        Rules {
            has_corners_win: true,
            has_edge_win: false,
            has_fort_win: true,
            has_shield_walls: true,
            initial_fen: "3aaaaa3/5a5/11/a4d4a/a3ddd3a/aa1ddkdd1aa/a3ddd3a/a4d4a/11/5a5/3aaaaa3 a".to_string(),
            is_king_strong: true
        }
    }

    pub fn create_historical_rules() -> Rules {
        Rules {
            has_corners_win: false,
            has_edge_win: true,
            has_fort_win: false,
            has_shield_walls: false,
            initial_fen: "4aaa4/4aaa4/5d5/5d5/aa3d3aa/aadddkdddaa/aa3d3aa/5d5/5d5/4aaa4/4aaa4 a".to_string(),
            is_king_strong: false
        }
    }
}