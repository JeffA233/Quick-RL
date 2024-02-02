use rlgym_sim_gym::gamestates::game_state::GameState;

// use rlgym_sim_gym::action_parser::ActionParser;
use rlgym_sim_gym::ActionParser;

/// Necto parser for Matrix
pub struct NectoAction {
    _lookup_table: Vec<[f32; 8]>,
}

impl Default for NectoAction {
    fn default() -> Self {
        Self::new()
    }
}

impl NectoAction {
    pub fn new() -> Self {
        NectoAction {
            _lookup_table: NectoAction::make_lookup_table(),
        }
    }

    fn make_lookup_table() -> Vec<[f32; 8]> {
        let mut actions = Vec::<[f32; 8]>::with_capacity(90);
        for throttle in [-1., 0., 1.] {
            for steer in [-1., 0., 1.] {
                for boost in [0., 1.] {
                    for handbrake in [0., 1.] {
                        if boost == 1. && throttle != 1. {
                            continue;
                        }
                        let part: f32 = if throttle != 0. {
                            throttle
                        } else if boost != 0. {
                            boost
                        } else {
                            0.
                        };
                        actions.push([part, steer, 0., steer, 0., 0., boost, handbrake]);
                    }
                }
            }
        }
        for pitch in [-1., 0., 1.] {
            for yaw in [-1., 0., 1.] {
                for roll in [-1., 0., 1.] {
                    for jump in [0., 1.] {
                        for boost in [0., 1.] {
                            if jump == 1. && yaw != 0. {
                                continue;
                            }
                            if pitch == roll && roll == jump && jump == 0. {
                                continue;
                            }
                            let handbrake = if jump == 1. && (pitch != 0. || yaw != 0. || roll != 0.) { 1. } else { 0. };
                            actions.push([boost, yaw, pitch, yaw, roll, jump, boost, handbrake]);
                        }
                    }
                }
            }
        }
        actions
    }
}

impl ActionParser for NectoAction {
    fn get_action_space(&mut self) -> Vec<usize> {
        vec![self._lookup_table.len()]
    }

    fn parse_actions(&mut self, actions: Vec<Vec<f32>>, _state: &GameState) -> Vec<Vec<f32>> {
        let mut parsed_actions = Vec::<Vec<f32>>::new();
        for action_vec in actions {
            for action in action_vec {
                parsed_actions.push(self._lookup_table[action as usize].to_vec());
            }
        }
        parsed_actions
    }
}
