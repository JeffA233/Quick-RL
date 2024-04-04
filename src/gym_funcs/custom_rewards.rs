use rlgym_sim_gym::{
    common_values::{BALL_MAX_SPEED, BLUE_GOAL_BACK, BLUE_TEAM, ORANGE_GOAL_BACK},
    gamestates::{
        game_state::GameState,
        physics_object::{Position, Velocity},
        player_data::PlayerData,
    },
};

use rlgym_sim_gym::{
    // combined_reward::CombinedReward,
    CombinedReward,
    // },
    // default_reward::RewardFn,
    RewardFn,
};

// use numpy::*;
// use ndarray::*;
use serde_json::from_reader;
// use itertools::Itertools;
use rustc_hash::FxHashMap as HashMap;
use std::{
    // collections::HashMap,
    fs::*,
    io::BufReader,
};

pub fn get_reward_json() -> Option<(Vec<f32>, Vec<bool>, f32)> {
    let file_str = "D:\\Users\\Jeffrey\\Documents\\GitHub\\RL AI bot\\rlgym_quickstart_tutorial_bot-main\\rlgym_rust\\reward_config.json";
    let file = File::open(file_str).unwrap();
    // let mut zip_file = ZipArchive::new(file).unwrap();
    // let zip_file = zip_file.by_index(0).unwrap();
    let reader = BufReader::new(file);
    let rew_setup: Result<(Vec<f32>, Vec<bool>, f32), serde_json::Error> = from_reader(reader);
    match rew_setup {
        Ok(values) => Some(values),
        // for now we want to panic as we don't want to use internal values
        Err(e) => panic!("err from reward func when loading json: {e}"),
        // Err(e) => {
        //     println!("err from reward func when loading json, rolling back to internal weights/bools: {e}");
        //     None
        // }
    }
}

pub fn get_rewards_and_weights() -> (Vec<Box<dyn RewardFn>>, Vec<f32>, Vec<bool>, f32) {
    let reward_fn_vec: Vec<Box<dyn RewardFn>> = vec![
        Box::new(GatherBoostRewardBasic::new()),
    ];

    // let op = get_reward_json();
    // we don't care about using the json for now, just make a simple test
    let op = None;
    let (weights, bools, team_spirit) = match op {
        Some(val) => val,
        None => {
            // fall back in case of testing
            // println!("Reverting to built-in rew weights");
            let team_spirit = 0.6;

            let weights = vec![1.0; reward_fn_vec.len()];
            assert!(weights.len() == reward_fn_vec.len());

            let reward_bools = vec![true; reward_fn_vec.len()];
            assert!(reward_bools.len() == reward_fn_vec.len());

            (weights, reward_bools, team_spirit)
        }
    };

    (reward_fn_vec, weights, bools, team_spirit)
}

/// returns configured custom rewards for Matrix usage, this part is meant only for the non-Rust multi-instance configuration
pub fn get_custom_reward_func_tester() -> Box<dyn RewardFn> {
    let (reward_fn_vec, weights, _reward_bools, _) = get_rewards_and_weights();

    Box::new(CombinedReward::new(reward_fn_vec, weights))
}

pub struct PositionTeamHolder {
    pub blue_team: [Position; 2],
    pub orange_team: [Position; 2],
}

pub struct VelocityTeamHolder {
    pub blue_team: [Velocity; 2],
    pub orange_team: [Velocity; 2],
}

pub struct BallAccelToGoal3 {
    pub last_ball_vel_to_targ: VelocityTeamHolder,
    pub last_ball_dist_to_targ: PositionTeamHolder,
    pub last_state_tick: u64,
    pub normalized_speed: f32,
}

impl BallAccelToGoal3 {
    pub fn new(normalized_speed_op: Option<f32>) -> Self {
        Self {
            // slot 1 is the current calculation, slot 0 is from the previous step
            last_ball_vel_to_targ: VelocityTeamHolder {
                blue_team: [Velocity {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                }; 2],
                orange_team: [Velocity {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                }; 2],
            },
            last_ball_dist_to_targ: PositionTeamHolder {
                blue_team: [Position {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                }; 2],
                orange_team: [Position {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                }; 2],
            },
            last_state_tick: 0,
            normalized_speed: normalized_speed_op.unwrap_or(BALL_MAX_SPEED),
        }
    }

    fn update_vel_and_dist(&mut self, state: &GameState, reset: bool) {
        if reset {
            // we can get a rough estimate of the last distance by rolling back 1/(120/tick_skip) * ball velocity
            let distance_to_subtract = state.ball.linear_velocity * (1. / 15.);
            let previous_position = state.ball.position - distance_to_subtract;
            let pos_diff = ORANGE_GOAL_BACK - previous_position;
            self.last_ball_dist_to_targ.blue_team[0] = Position {
                x: pos_diff.x.abs(),
                y: pos_diff.y.abs(),
                z: pos_diff.z.abs(),
            };
            let pos_diff = BLUE_GOAL_BACK - previous_position;
            self.last_ball_dist_to_targ.orange_team[0] = Position {
                x: pos_diff.x.abs(),
                y: pos_diff.y.abs(),
                z: pos_diff.z.abs(),
            };
        } else {
            // move previous measurements to last slot
            self.last_ball_dist_to_targ.blue_team[0] = self.last_ball_dist_to_targ.blue_team[1];
            self.last_ball_dist_to_targ.orange_team[0] = self.last_ball_dist_to_targ.orange_team[1];

            self.last_ball_vel_to_targ.blue_team[0] = self.last_ball_vel_to_targ.blue_team[1];
            self.last_ball_vel_to_targ.orange_team[0] = self.last_ball_vel_to_targ.orange_team[1];
        }

        // update position difference
        let pos_diff = ORANGE_GOAL_BACK - state.ball.position;
        self.last_ball_dist_to_targ.blue_team[1] = Position {
            x: pos_diff.x.abs(),
            y: pos_diff.y.abs(),
            z: pos_diff.z.abs(),
        };
        let pos_diff = BLUE_GOAL_BACK - state.ball.position;
        self.last_ball_dist_to_targ.orange_team[1] = Position {
            x: pos_diff.x.abs(),
            y: pos_diff.y.abs(),
            z: pos_diff.z.abs(),
        };

        // update velocity to objective
        self.last_ball_vel_to_targ.blue_team[1] = (self.last_ball_dist_to_targ.blue_team[0]
            - self.last_ball_dist_to_targ.blue_team[1])
            .into();
        // multiply by 15 (120/tick_skip) to get actual velocity in uu/s
        self.last_ball_vel_to_targ.blue_team[1] = self.last_ball_vel_to_targ.blue_team[1] * 15.;
        self.last_ball_vel_to_targ.orange_team[1] = (self.last_ball_dist_to_targ.orange_team[0]
            - self.last_ball_dist_to_targ.orange_team[1])
            .into();
        self.last_ball_vel_to_targ.orange_team[1] = self.last_ball_vel_to_targ.orange_team[1] * 15.;
    }
}

impl Default for BallAccelToGoal3 {
    fn default() -> Self {
        Self::new(None)
    }
}

impl RewardFn for BallAccelToGoal3 {
    fn reset(&mut self, initial_state: &GameState) {
        self.update_vel_and_dist(initial_state, true);

        // set these to 0 to force an update on the next get_reward call
        self.last_state_tick = 0;
    }

    fn get_reward(&mut self, player: &PlayerData, state: &GameState) -> f32 {
        if self.last_state_tick != state.tick_num {
            self.update_vel_and_dist(state, false);

            self.last_state_tick = state.tick_num;
        }

        if player.ball_touched {
            let ret = if player.team_num == BLUE_TEAM {
                (self.last_ball_vel_to_targ.blue_team[1] - self.last_ball_vel_to_targ.blue_team[0])
                    .divide_by_var(self.normalized_speed)
                    .into_array()
                    .iter()
                    .sum::<f32>()
                    * 4.0
            } else {
                (self.last_ball_vel_to_targ.orange_team[1]
                    - self.last_ball_vel_to_targ.orange_team[0])
                    .divide_by_var(self.normalized_speed)
                    .into_array()
                    .iter()
                    .sum::<f32>()
                    * 4.0
            };
            if ret > 0. {
                ret
            } else {
                0.
            }
        } else {
            0.
        }
    }

    fn get_final_reward(&mut self, player: &PlayerData, state: &GameState) -> f32 {
        self.get_reward(player, state)
    }
}

/// reward for having jump available for use
#[derive(Default)]
pub struct JumpReward {}

impl RewardFn for JumpReward {
    fn reset(&mut self, _initial_state: &GameState) {}

    fn get_reward(&mut self, player: &PlayerData, _state: &GameState) -> f32 {
        player.has_jump as i32 as f32
    }

    fn get_final_reward(&mut self, player: &PlayerData, state: &GameState) -> f32 {
        self.get_reward(player, state)
    }
}

/// trial reward for trying to reduce initial bad behavior of constant dodging
pub struct FlipPunish {
    timer: u64,
    previous_state: bool,
    flip_count: u64,
    flip_threshold: u64,
    time_threshold: u64,
}

impl FlipPunish {
    pub fn new(flip_threshold_val: Option<u64>, time_threshold_val: Option<u64>) -> Self {
        let flip_threshold = flip_threshold_val.unwrap_or(5);
        let time_threshold = time_threshold_val.unwrap_or(10);
        FlipPunish {
            timer: 0,
            previous_state: false,
            flip_count: 0,
            flip_threshold,
            time_threshold,
        }
    }
}

impl RewardFn for FlipPunish {
    fn reset(&mut self, _initial_state: &GameState) {
        self.flip_count = 0;
        self.timer = 0;
        self.previous_state = true;
    }

    fn get_reward(&mut self, player: &PlayerData, _state: &GameState) -> f32 {
        if (player.last_actions.jump) != self.previous_state {
            self.previous_state = player.last_actions.jump;
            if self.timer < 3 {
                self.timer = 0;
                // self.flip_count = 0;
            } else {
                self.timer = 0;
                self.flip_count += 1;
            }
            if self.flip_count >= self.flip_threshold {
                return -1.;
            // } else if self.flip_count >= self.flip_threshold {
            //     out = -(self.flip_count as f32 / self.flip_threshold as f32);
            } else {
                return 0.;
            }
        } else {
            self.timer += 1;
            if self.timer > self.time_threshold {
                self.flip_count = 0;
            }
        }
        0.
    }

    fn get_final_reward(&mut self, player: &PlayerData, state: &GameState) -> f32 {
        self.get_reward(player, state)
    }
}

/// reward for gathering boost, based on boost amount instead of collecting pads
pub struct GatherBoostRewardBasic {
    last_boost: HashMap<i32, f32>,
    player_episode_boost_total: HashMap<i32, f32>,
    avg_ep_gather_ratio: f32,
}

impl GatherBoostRewardBasic {
    pub fn new() -> Self {
        let mut hashmap = HashMap::default();
        let mut player_ep_hashmap = HashMap::default();
        for i in 0..6 {
            hashmap.insert(i, 34.);
        }
        for i in 0..6 {
            player_ep_hashmap.insert(i, 0.);
        }
        Self {
            last_boost: hashmap,
            player_episode_boost_total: player_ep_hashmap,
            avg_ep_gather_ratio: 1.,
        }
    }
}

impl RewardFn for GatherBoostRewardBasic {
    fn reset(&mut self, initial_state: &GameState) {
        for player in &initial_state.players {
            self.last_boost.insert(player.car_id, player.boost_amount);
        }
    }

    fn get_reward(&mut self, player: &PlayerData, _state: &GameState) -> f32 {
        let last_boost_op = self.last_boost.insert(player.car_id, player.boost_amount);
        // in case player.car_id wasn't populated
        let last_boost = last_boost_op.unwrap_or(0.);
        let mut boost_differential;
        if player.boost_amount > last_boost {
            boost_differential = player.boost_amount - last_boost;
            let player_ep_total_op = self.player_episode_boost_total.get_mut(&player.car_id);
            // in case player.car_id wasn't populated
            match player_ep_total_op {
                Some(val) => *val += boost_differential,
                None => {
                    self.player_episode_boost_total
                        .insert(player.car_id, boost_differential);
                }
            };
            boost_differential *= self.avg_ep_gather_ratio;
        } else {
            boost_differential = 0.;
        }
        boost_differential
    }

    fn get_final_reward(&mut self, player: &PlayerData, state: &GameState) -> f32 {
        self.get_reward(player, state)
    }
}

pub struct EventRewardTeamGoal {
    weight: f32,
    last_registered_value_blue: i32,
    last_registered_value_orange: i32,
}

impl EventRewardTeamGoal {
    pub fn new(team_goal: f32) -> EventRewardTeamGoal {
        EventRewardTeamGoal {
            weight: team_goal,
            last_registered_value_blue: 0,
            last_registered_value_orange: 0,
        }
    }
}

impl RewardFn for EventRewardTeamGoal {
    fn reset(&mut self, initial_state: &GameState) {
        self.last_registered_value_blue = initial_state.blue_score;
        self.last_registered_value_orange = initial_state.orange_score;
    }

    fn get_reward(&mut self, player: &PlayerData, state: &GameState) -> f32 {
        let ret = if player.team_num == BLUE_TEAM {
            state.blue_score - self.last_registered_value_blue
        } else {
            state.orange_score - self.last_registered_value_orange
        };

        // muh no branch code go brrrr
        // let mut ret = (player.team_num == BLUE_TEAM) as i32 * (curr_val - self.last_registered_value_blue);
        // ret += (player.team_num == ORANGE_TEAM) as i32 * (curr_val - self.last_registered_value_orange);

        ret as f32 * self.weight
    }

    fn get_final_reward(&mut self, player: &PlayerData, state: &GameState) -> f32 {
        self.get_reward(player, state)
    }
}

pub struct EventRewardConcede {
    weight: f32,
    last_registered_value_blue: i32,
    last_registered_value_orange: i32,
}

impl EventRewardConcede {
    pub fn new(concede: f32) -> EventRewardConcede {
        EventRewardConcede {
            weight: concede,
            last_registered_value_blue: 0,
            last_registered_value_orange: 0,
        }
    }
}

impl RewardFn for EventRewardConcede {
    fn reset(&mut self, initial_state: &GameState) {
        self.last_registered_value_orange = initial_state.orange_score;
        self.last_registered_value_blue = initial_state.blue_score;
    }

    fn get_reward(&mut self, player: &PlayerData, state: &GameState) -> f32 {
        let ret = if player.team_num == BLUE_TEAM {
            state.orange_score - self.last_registered_value_orange
        } else {
            state.blue_score - self.last_registered_value_blue
        };

        ret as f32 * self.weight
    }

    fn get_final_reward(&mut self, player: &PlayerData, state: &GameState) -> f32 {
        self.get_reward(player, state)
    }
}

// /// "Wrapper" that collects a set of boxed reward functions and iterates through them to get a single float.
// /// Has other functionality including reward logging. Unlike the mult-instance version, it writes on its own new thread.
// pub struct SB3CombinedLogReward {
//     reward_file_path: PathBuf,
//     // reward_file: String,
//     // lockfile: String,
//     final_mult: f32,
//     returns: Vec<f32>,
//     combined_reward_fns: Vec<Box<dyn RewardFn>>,
//     combined_reward_weights: Vec<f32>,
// }

// impl SB3CombinedLogReward {
//     fn new(
//         reward_structs: Vec<Box<dyn RewardFn>>,
//         reward_weights: Vec<f32>,
//         file_location: Option<String>,
//         final_mult: Option<f32>,
//     ) -> Self {
//         let file_location = match file_location {
//             Some(file_location) => file_location,
//             None => "./combinedlogfiles".to_owned(),
//         };

//         let reward_file = format!("{}/rewards_test.txt", file_location);
//         let reward_file_path = Path::new(&reward_file);
//         // let lockfile = format!("{}/reward_lock", file_location);

//         let final_mult = final_mult.unwrap_or(1.);
//         let exists = Path::new(&file_location).exists();
//         if !exists {
//             let res = create_dir(&file_location);
//             match res {
//                 Err(error) => {
//                     if error.kind() == AlreadyExists {
//                     } else {
//                         panic!("{error}")
//                     }
//                 }
//                 Ok(out) => out,
//             }
//         }
//         for i in 0..100 {
//             if i == 99 {
//                 panic!("too many attempts taken to lock the file in new")
//             }

//             let out = OpenOptions::new()
//                 .create(true)
//                 .write(true)
//                 .truncate(true)
//                 .open(reward_file_path);

//             let file = match out {
//                 Err(out) => {
//                     if out.kind() == PermissionDenied {
//                         continue;
//                     } else {
//                         println!("{out}");
//                         continue;
//                     }
//                 }
//                 Ok(_file) => _file,
//             };

//             let out = file.lock_exclusive();

//             match out {
//                 // Err(out) => {if out.kind() == PermissionDenied {continue} else {continue}},
//                 Err(_) => continue,
//                 Ok(_file) => _file,
//             };

//             file.unlock().unwrap();
//             break;
//         }

//         SB3CombinedLogReward {
//             reward_file_path: reward_file_path.to_owned(),
//             // reward_file: reward_file,
//             // lockfile: lockfile,
//             final_mult,
//             returns: vec![0.; reward_structs.len()],
//             combined_reward_fns: reward_structs,
//             combined_reward_weights: reward_weights,
//         }
//     }
// }

// impl RewardFn for SB3CombinedLogReward {
//     fn reset(&mut self, _initial_state: &GameState) {
//         // self.returns = vec![0.; self.combined_reward_fns.len()];
//         for func in &mut self.combined_reward_fns {
//             func.reset(_initial_state);
//         }
//         self.returns.fill(0.);
//     }

//     fn get_reward(&mut self, player: &PlayerData, state: &GameState) -> f32 {
//         let mut rewards = Vec::<f32>::new();

//         for func in &mut self.combined_reward_fns {
//             let val = func.get_reward(player, state);
//             rewards.push(val);
//         }

//         let vals = element_mult_vec(&rewards, &self.combined_reward_weights);
//         self.returns = element_add_vec(&self.returns, &vals);
//         let sum = vals.iter().sum::<f32>();
//         sum * self.final_mult
//     }

//     fn get_final_reward(&mut self, player: &PlayerData, state: &GameState) -> f32 {
//         let mut rewards = Vec::<f32>::new();

//         for func in &mut self.combined_reward_fns {
//             let val = func.get_final_reward(player, state);
//             rewards.push(val);
//         }

//         let vals = element_mult_vec(&rewards, &self.combined_reward_weights);
//         // self.returns = element_add_vec(&self.returns, &vals);
//         let local_ret = element_add_vec(&self.returns, &vals);

//         // let local_ret = self.returns.clone();
//         self.returns.fill(0.);
//         let reward_file = self.reward_file_path.clone();

//         thread::spawn(move || file_put(local_ret, reward_file.as_path()));

//         let sum = vals.iter().sum::<f32>();
//         sum * self.final_mult
//     }
// }

// /// used to write the rewards to a file on a separate thread from the reward fn
// fn file_put(returns_local: Vec<f32>, reward_file: &Path) {
//     for i in 0..100 {
//         if i == 99 {
//             panic!("too many attempts taken to lock the file in file_put")
//         }
//         let out = OpenOptions::new().append(true).read(true).open(reward_file);

//         let file = match out {
//             Err(out) => {
//                 println!("file error: {out}");
//                 // if out.kind() == PermissionDenied {continue} else {continue};},
//                 continue;
//             }
//             Ok(_file) => _file,
//         };

//         let out = file.lock_exclusive();

//         match out {
//             Err(out) => {
//                 println!("lock error: {out}");
//                 // if out.kind() == PermissionDenied {continue} else {continue};
//                 continue;
//             }
//             Ok(_file) => _file,
//         };

//         let mut buf = BufWriter::new(&file);

//         let mut string = String::new();
//         string += "[";
//         for ret in returns_local.iter().take(returns_local.len() - 1) {
//             string = string + &format!("{}, ", ret)
//         }
//         string = string + &format!("{}]", returns_local[returns_local.len() - 1]);
//         writeln!(&mut buf, "{}", string).unwrap();
//         let out = buf.flush();
//         match out {
//             Ok(out) => out,
//             Err(err) => println!("buf.flush in logger failed with error: {err}"),
//         };
//         file.unlock().unwrap();
//         break;
//     }
// }
