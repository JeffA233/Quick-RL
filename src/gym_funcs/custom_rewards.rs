use rlgym_sim_gym::{
    common_values::{BALL_MAX_SPEED, BLUE_GOAL_BACK, BLUE_TEAM, ORANGE_GOAL_BACK, ORANGE_TEAM},
    gamestates::{
        game_state::GameState,
        physics_object::{Position, Velocity},
        player_data::PlayerData,
    },
    math::{element_add_vec, element_mult_vec},
};

use rlgym_sim_gym::{
    ball_goal_rewards::VelocityBallToGoalReward,
    // common_rewards::{
    misc_rewards::{EventReward, SaveBoostReward, VelocityReward},
    player_ball_rewards::VelocityPlayerToBallReward,
    // combined_reward::CombinedReward,
    CombinedReward,
    // },
    // default_reward::RewardFn,
    RewardFn,
};

// use numpy::*;
// use ndarray::*;
use crossbeam_channel::Sender;
use serde_json::from_reader;
// use itertools::Itertools;
use rustc_hash::FxHashMap as HashMap;
use std::{
    // collections::HashMap,
    fs::*,
    io::BufReader,
    path::PathBuf,
};
// use itertools::min;
use std::io::ErrorKind::*;
use std::io::{BufWriter, Write};
// use std::fs::File;
use fs2::FileExt;
use std::path::Path;
use std::thread;
// use rayon::prelude::*;

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
    // let dribble_reward: Box<DribbleAirTouchReward2> = Box::new(DribbleAirTouchReward2::new(Some(180.), Some(1.), Some(5.), Some(20)));

    let reward_fn_vec: Vec<Box<dyn RewardFn>> = vec![
        // Box::new(VelocityPlayerToBallReward::new(None)),
        // // Box::new(VelBallToGoalWeightedAerialReward::new(None, None, None, None, None, None, None, None, None, None)),
        // Box::new(VelocityBallToGoalReward::new(None, None)),
        // Box::new(AgentOvercommitFuncReward::new(Some(400.), None)),
        // Box::new(GatherBoostReward::new(Some(0.022))),
        // Box::new(SaveBoostReward::new()),
        // Box::new(LeftKickoffReward::new()),
        // Box::new(VelocityReward::new(None)),
        // Box::new(EventReward::new(None, None, None, None, Some(5.), Some(15.), Some(10.), None)),
        // Box::new(EventRewardTeamGoal::new(35.)),
        // Box::new(EventRewardConcede::new(-35.)),
        // Box::<JumpReward>::default(),
        Box::new(GatherBoostRewardBasic::new(None)),
        // Box::new(AerialWeightedWrapper::new(dribble_reward, Some(180.), Some(700.), None, Some(4.0))),
        // Box::new(FlipPunish::new(None, None)),
        // Box::new(BallAccelToGoal3::new(None)),
        // Box::<BallHeightInfo>::default(),
        // Box::<BallHeightTouchInfo>::default(),
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
            // let weights = vec![0.22, 2.5, 0.4, 10.0, 0.11, 7.0, 0.01, 1.0, 1.0, 1.0, 0.002, 2.0, 0.5, 7.5, 1., 1.];
            // let weights = vec![0.11];
            // let weights = vec![1.0];
            let weights = vec![1.0; reward_fn_vec.len()];
            assert!(weights.len() == reward_fn_vec.len());

            // let reward_bools = vec![false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false];
            // let reward_bools = vec![true];
            let reward_bools = vec![true; reward_fn_vec.len()];
            assert!(reward_bools.len() == reward_fn_vec.len());
            (weights, reward_bools, team_spirit)
        }
    };

    (reward_fn_vec, weights, bools, team_spirit)
}

/// returns configured custom rewards for Matrix usage, this part is meant only for the non-Rust multi-instance configuration
// pub fn get_custom_reward_func() -> Box<dyn RewardFn> {
//     let (reward_fn_vec, weights, _reward_bools, _) = get_rewards_and_weights();

//     Box::new(SB3CombinedLogReward::new(
//         reward_fn_vec,
//         weights,
//         Some(r"F:\Users\Jeffrey\AppData\Local\Temp".to_string()),
//         Some(0.1),
//     ))
// }

/// returns configured custom rewards for Matrix usage, this part is meant only for the non-Rust multi-instance configuration
pub fn get_custom_reward_func_tester() -> Box<dyn RewardFn> {
    let (reward_fn_vec, weights, _reward_bools, _) = get_rewards_and_weights();

    Box::new(CombinedReward::new(reward_fn_vec, weights))
}

/// returns configured custom rewards for Matrix usage, built for Rust multi-instance
// pub fn get_custom_reward_func_mult_inst(reward_send_chan: Sender<Vec<f32>>) -> Box<dyn RewardFn> {
//     let (reward_fn_vec, weights, reward_bools, team_spirit) = get_rewards_and_weights();

//     // Box::new(SB3CombinedLogRewardMultInst::new(
//     //     reward_fn_vec,
//     //     weights,
//     //     reward_bools,
//     //     Some(r"F:\Users\Jeffrey\AppData\Local\Temp".to_string()),
//     //     Some(0.1),
//     //     reward_send_chan,
//     // ))

//     Box::new(SB3CombinedLogRewardMultInstTeam::new(
//         reward_fn_vec,
//         weights,
//         reward_bools,
//         Some(r"F:\Users\Jeffrey\AppData\Local\Temp".to_string()),
//         Some(0.5),
//         reward_send_chan,
//         team_spirit,
//     ))
// }

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

/// reward only for the agent that is on the left of their respective side (meant to try to be the normal "left goes for kickoff" rule)
pub struct LeftKickoffReward {
    vel_dir_reward: VelocityPlayerToBallReward,
    kickoff_id_orange: i32,
    kickoff_id_blue: i32,
}

impl LeftKickoffReward {
    pub fn new() -> Self {
        LeftKickoffReward {
            vel_dir_reward: VelocityPlayerToBallReward::new(None),
            kickoff_id_orange: -1,
            kickoff_id_blue: -1,
        }
    }
}

impl Default for LeftKickoffReward {
    fn default() -> Self {
        LeftKickoffReward {
            vel_dir_reward: VelocityPlayerToBallReward::new(None),
            kickoff_id_orange: -1,
            kickoff_id_blue: -1,
        }
    }
}

impl RewardFn for LeftKickoffReward {
    fn reset(&mut self, _initial_state: &GameState) {
        self.vel_dir_reward.reset(_initial_state);
    }

    fn get_reward(&mut self, player: &PlayerData, state: &GameState) -> f32 {
        // only should occur during kickoff or an extremely rare case in a game
        if state.ball.position.x == 0. && state.ball.position.y == 0. {
            // check if we need to calculate the kickoff ids
            if self.kickoff_id_blue == -1 || self.kickoff_id_orange == -1 {
                // 1s shortcut since 1s doesn't require sorting/checking
                // FIXME: self_play=false doesn't occur for now and isn't covered but should be considered in the future
                if state.players.len() <= 3 {
                    for car in &state.players {
                        if car.team_num == BLUE_TEAM {
                            self.kickoff_id_blue = car.car_id;
                        } else {
                            self.kickoff_id_orange = car.car_id;
                        }
                    }
                // 2s and 3s calc
                } else {
                    let mut blue_car = &state.players[0];
                    let mut orange_car = &state.players[1];

                    let mut blue_car_found = false;
                    let mut orange_car_found = false;
                    for car in &state.players {
                        // find a car on each team to compare against
                        if car.team_num == BLUE_TEAM && !blue_car_found {
                            blue_car = car;
                            blue_car_found = true;
                        } else if car.team_num == ORANGE_TEAM && !orange_car_found {
                            orange_car = car;
                            orange_car_found = true;
                        }
                        if blue_car_found && orange_car_found {
                            break;
                        }
                    }

                    for car in &state.players {
                        if car.team_num == BLUE_TEAM {
                            if car.car_id == blue_car.car_id {
                                continue;
                            }
                            if car.car_data.position.y >= blue_car.car_data.position.y
                                && car.car_data.position.x > blue_car.car_data.position.x
                            {
                                blue_car = car;
                            }
                        } else {
                            if car.car_id == orange_car.car_id {
                                continue;
                            }
                            if car.inverted_car_data.position.y
                                >= orange_car.inverted_car_data.position.y
                                && car.inverted_car_data.position.x
                                    > orange_car.inverted_car_data.position.x
                            {
                                orange_car = car;
                            }
                        }
                    }
                    self.kickoff_id_blue = blue_car.car_id;
                    self.kickoff_id_orange = orange_car.car_id;
                }
            }

            if player.car_id == self.kickoff_id_orange || player.car_id == self.kickoff_id_blue {
                let vel_dir_rew = self.vel_dir_reward.get_reward(player, state);
                // let rew_signum = vel_dir_rew.signum();
                if vel_dir_rew < 0. {
                    vel_dir_rew * 2.
                } else {
                    vel_dir_rew.powi(2) * vel_dir_rew.signum()
                }
            } else {
                0.
            }
        } else {
            self.kickoff_id_blue = -1;
            self.kickoff_id_orange = -1;
            0.
        }
    }

    fn get_final_reward(&mut self, player: &PlayerData, state: &GameState) -> f32 {
        self.get_reward(player, state)
    }
}

/// reward for gathering boost, based on boost amount instead of collecting pads
pub struct GatherBoostReward {
    last_boost: HashMap<i32, f32>,
    player_episode_boost_total: HashMap<i32, f32>,
    target_ep_gather_rate: f32,
    avg_ep_gather_ratio: f32,
    last_episode_tick_start: u64,
}

impl GatherBoostReward {
    pub fn new(target_boost_rate: Option<f32>) -> Self {
        let target_ep_gather_rate = target_boost_rate.unwrap_or(0.005);
        let mut hashmap = HashMap::default();
        let mut player_ep_hashmap = HashMap::default();
        for i in 0..6 {
            hashmap.insert(i, 34.);
        }
        for i in 0..6 {
            player_ep_hashmap.insert(i, 0.);
        }
        GatherBoostReward {
            last_boost: hashmap,
            player_episode_boost_total: player_ep_hashmap,
            target_ep_gather_rate,
            avg_ep_gather_ratio: 1.,
            last_episode_tick_start: 0,
        }
    }
}

impl RewardFn for GatherBoostReward {
    fn reset(&mut self, initial_state: &GameState) {
        for player in &initial_state.players {
            self.last_boost.insert(player.car_id, player.boost_amount);
        }
        // update boost per tick here
        // let mut values_total = 0.;
        // for values in self.player_episode_boost_total.values() {
        //     values_total += values;
        // }
        let values_total: f32 = self.player_episode_boost_total.values().sum();

        if values_total < 0.1 {
            self.avg_ep_gather_ratio = 1.5;
        } else {
            let num_players = self.player_episode_boost_total.len();
            let avg_boost_totals = values_total / num_players as f32;
            let delta = initial_state.tick_num - self.last_episode_tick_start;
            let boost_per_tick = avg_boost_totals / delta as f32;
            // if boost_per_tick >= self.target_ep_gather_rate {
            //     self.avg_ep_gather_ratio = 1.;
            // } else {
            // self.avg_ep_gather_ratio = (2. - boost_per_tick / self.target_ep_gather_rate).ln() * 4.;
            // (state.ball.position.z - self.target_height + (self.target_height / 4.)) / (self.target_height / 4.).powf(1.25);
            self.avg_ep_gather_ratio =
                (self.target_ep_gather_rate - boost_per_tick + 0.002) / (0.002);

            if self.avg_ep_gather_ratio < 0.5 {
                self.avg_ep_gather_ratio = 0.5;
            } else if self.avg_ep_gather_ratio > 1.5 {
                self.avg_ep_gather_ratio = 1.5;
            }
            // }
            // if self.avg_ep_gather_ratio.is_nan() || self.avg_ep_gather_ratio > 6. {
            //     let placeholder = self.avg_ep_gather_ratio;
            //     panic!("caught a bad number, num_players: {num_players}, avg_boost_totals: {avg_boost_totals}, delta: {delta}, boost_per_tick: {boost_per_tick}, avg_ep_gather_ratio: {placeholder}")
            // }
        }

        // if self.avg_ep_gather_ratio.is_nan() || self.avg_ep_gather_ratio > 6. {
        //     let placeholder = self.avg_ep_gather_ratio;
        //     panic!("caught a bad number, avg_ep_gather_ratio: {placeholder}")
        // }

        self.last_episode_tick_start = initial_state.tick_num;
        self.player_episode_boost_total.clear();
        for player in &initial_state.players {
            self.player_episode_boost_total.insert(player.car_id, 0.);
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
            // if boost_differential > 6. || boost_differential.is_nan() {
            //     let avg_ep_gather_ratio = self.avg_ep_gather_ratio;
            //     panic!("caught a bad number, boost_differental: {boost_differential}, avg_ep_gather_ratio: {avg_ep_gather_ratio}")
            // }
        } else {
            boost_differential = 0.;
        }
        boost_differential
    }

    fn get_final_reward(&mut self, player: &PlayerData, state: &GameState) -> f32 {
        self.get_reward(player, state)
    }
}

/// reward for gathering boost, based on boost amount instead of collecting pads
pub struct GatherBoostRewardBasic {
    last_boost: HashMap<i32, f32>,
    player_episode_boost_total: HashMap<i32, f32>,
    target_ep_gather_rate: f32,
    avg_ep_gather_ratio: f32,
    last_episode_tick_start: u64,
}

impl GatherBoostRewardBasic {
    pub fn new(target_boost_rate: Option<f32>) -> Self {
        let target_ep_gather_rate = target_boost_rate.unwrap_or(0.005);
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
            target_ep_gather_rate,
            avg_ep_gather_ratio: 1.,
            last_episode_tick_start: 0,
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
            // if boost_differential > 6. || boost_differential.is_nan() {
            //     let avg_ep_gather_ratio = self.avg_ep_gather_ratio;
            //     panic!("caught a bad number, boost_differental: {boost_differential}, avg_ep_gather_ratio: {avg_ep_gather_ratio}")
            // }
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

    // fn _extract_values(player: &PlayerData, state: &GameState) -> i32 {
    //     if player.team_num == BLUE_TEAM {
    //         state.blue_score
    //     } else {
    //         state.orange_score
    //     }
    // }
}

impl RewardFn for EventRewardTeamGoal {
    fn reset(&mut self, initial_state: &GameState) {
        self.last_registered_value_blue = initial_state.blue_score;
        self.last_registered_value_orange = initial_state.orange_score;
    }

    fn get_reward(&mut self, player: &PlayerData, state: &GameState) -> f32 {
        // let curr_val = EventRewardTeamGoal::_extract_values(player, state);
        let ret = if player.team_num == BLUE_TEAM {
            // let curr_val = state.blue_score;
            state.blue_score - self.last_registered_value_blue
            // dbg
            // let val = curr_val - self.last_registered_value_blue;
            // assert!(val < 2, "got a value higher than 1 in EventRewardTeamGoal");
            // val
        } else {
            // let curr_val = state.orange_score;
            state.orange_score - self.last_registered_value_orange
            // dbg
            // let val = curr_val - self.last_registered_value_orange;
            // assert!(val < 2, "got a value higher than 1 in EventRewardTeamGoal");
            // val
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

    // fn _extract_values(player: &PlayerData, state: &GameState) -> i32 {
    //     if player.team_num == BLUE_TEAM {
    //         state.orange_score
    //     } else {
    //         state.blue_score
    //     }
    // }
}

impl RewardFn for EventRewardConcede {
    fn reset(&mut self, initial_state: &GameState) {
        self.last_registered_value_orange = initial_state.orange_score;
        self.last_registered_value_blue = initial_state.blue_score;
    }

    fn get_reward(&mut self, player: &PlayerData, state: &GameState) -> f32 {
        // let curr_val = EventRewardConcede::_extract_values(player, state);
        let ret = if player.team_num == BLUE_TEAM {
            // let curr_val = state.orange_score;
            state.orange_score - self.last_registered_value_orange
            // dbg
            // let val = curr_val - self.last_registered_value_orange;
            // assert!(val < 2, "got a value higher than 1 in EventRewardConcede");
            // val
        } else {
            // let curr_val = state.blue_score;
            state.blue_score - self.last_registered_value_blue
            // dbg
            // let val = curr_val - self.last_registered_value_blue;
            // assert!(val < 2, "got a value higher than 1 in EventRewardConcede");
            // val
        };

        ret as f32 * self.weight
    }

    fn get_final_reward(&mut self, player: &PlayerData, state: &GameState) -> f32 {
        self.get_reward(player, state)
    }
}

/// "Wrapper" that collects a set of boxed reward functions and iterates through them to get a single float.
/// Has other functionality including reward logging that sends info to a separate singular thread which writes for all instances
/// instead of each instance having its own writer
///
/// NOTE: This only works with at least one agent per team.
/// TODO: Normalize reward logging based on number of steps?
pub struct SB3CombinedLogRewardMultInstTeam {
    reward_sender: Sender<Vec<f32>>,
    final_mult: f32,
    returns: Vec<Vec<f32>>,
    combined_reward_fns: Vec<Box<dyn RewardFn>>,
    combined_reward_weights: Vec<f32>,
    log_only_reward: Vec<bool>,
    orange_team_vals: HashMap<i32, f32>,
    blue_team_vals: HashMap<i32, f32>,
    last_tick_calculated: u64,
    team_spirit: f32,
}

impl SB3CombinedLogRewardMultInstTeam {
    fn new(
        reward_structs: Vec<Box<dyn RewardFn>>,
        reward_weights: Vec<f32>,
        log_only_vec: Vec<bool>,
        _file_location: Option<String>,
        final_mult: Option<f32>,
        sender: Sender<Vec<f32>>,
        team_spirit: f32,
    ) -> Self {
        let final_mult = final_mult.unwrap_or(1.);

        let returns = vec![vec![0.; reward_structs.len()]; 8];

        SB3CombinedLogRewardMultInstTeam {
            reward_sender: sender,
            final_mult,
            returns,
            combined_reward_fns: reward_structs,
            combined_reward_weights: reward_weights,
            log_only_reward: log_only_vec,
            orange_team_vals: HashMap::default(),
            blue_team_vals: HashMap::default(),
            last_tick_calculated: 0,
            team_spirit,
        }
    }
}

impl RewardFn for SB3CombinedLogRewardMultInstTeam {
    fn reset(&mut self, _initial_state: &GameState) {
        for func in &mut self.combined_reward_fns {
            func.reset(_initial_state);
        }
        for returns in &mut self.returns {
            returns.fill(0.);
        }
        self.last_tick_calculated = 0;
        self.blue_team_vals.clear();
        self.orange_team_vals.clear();
    }

    fn pre_step(&mut self, _state: &GameState) {
        self.last_tick_calculated = 0;
    }

    fn get_reward(&mut self, player: &PlayerData, state: &GameState) -> f32 {
        if self.last_tick_calculated != state.tick_num {
            // calculate individual rewards
            for agent in &state.players {
                let final_val: f32 = self
                    .combined_reward_fns
                    .iter_mut()
                    .zip(&mut self.returns[agent.car_id as usize])
                    .zip(&self.combined_reward_weights)
                    .zip(&self.log_only_reward)
                    .map(|(((func, ret), weight), reward_bool)| {
                        // let agent_act = agent.last_actions;
                        // CarControls {
                        //     throttle: action[0],
                        //     steer: action[1],
                        //     pitch: action[2],
                        //     yaw: action[3],
                        //     roll: action[4],
                        //     jump: action[5] > 0.,
                        //     boost: action[6] > 0.,
                        //     handbrake: action[7] > 0.,
                        // },
                        // let previous_act = [
                        //     agent_act.throttle,
                        //     agent_act.steer,
                        //     agent_act.pitch,
                        //     agent_act.yaw,
                        //     agent_act.roll,
                        //     agent_act.jump as i32 as f32,
                        //     agent_act.boost as i32 as f32,
                        //     agent_act.handbrake as i32 as f32
                        // ];
                        let val = func.get_reward(agent, state);
                        let reward = val * *weight;

                        // add to self.returns for logging
                        *ret += reward;
                        // check if only logging
                        if *reward_bool {
                            reward
                        } else {
                            0.0
                        }
                    })
                    .sum();

                if agent.team_num == BLUE_TEAM {
                    self.blue_team_vals
                        .insert(agent.car_id, final_val * self.final_mult);
                } else {
                    self.orange_team_vals
                        .insert(agent.car_id, final_val * self.final_mult);
                }
            }

            // calculate team spirit rewards
            let blue_team_val_sum = self.blue_team_vals.values().sum::<f32>();
            let blue_team_val_mean = blue_team_val_sum / self.blue_team_vals.len() as f32;

            let orange_team_val_sum = self.orange_team_vals.values().sum::<f32>();
            let orange_team_val_mean = orange_team_val_sum / self.orange_team_vals.len() as f32;

            // let mut dbg_val = 0.;
            // blue team spirit calc
            for value in self.blue_team_vals.values_mut() {
                *value = ((1. - self.team_spirit) * *value)
                    + (self.team_spirit * blue_team_val_mean)
                    - orange_team_val_mean;
                assert!(
                    !value.is_nan() || !value.is_infinite(),
                    "found NaN or inf in combined team rew blue: {value}"
                );
                // dbg_val += *value;
            }

            // orange team spirit calc
            for value in self.orange_team_vals.values_mut() {
                *value = ((1. - self.team_spirit) * *value)
                    + (self.team_spirit * orange_team_val_mean)
                    - blue_team_val_mean;
                assert!(
                    !value.is_nan() || !value.is_infinite(),
                    "found NaN or inf in combined team rew orange: {value}"
                );
                // dbg_val += *value;
            }

            // assert!(dbg_val < 0.01, "dbg_val was too big, {dbg_val}, blue_team_vals: {:?}, orange_team_vals: {:?}", self.blue_team_vals, self.orange_team_vals);

            self.last_tick_calculated = state.tick_num;
        }

        if player.team_num == BLUE_TEAM {
            *self.blue_team_vals.get(&player.car_id).unwrap()
        } else {
            *self.orange_team_vals.get(&player.car_id).unwrap()
        }
    }

    fn get_final_reward(&mut self, player: &PlayerData, state: &GameState) -> f32 {
        if self.last_tick_calculated != state.tick_num {
            // calculate individual rewards
            for agent in &state.players {
                let final_val: f32 = self
                    .combined_reward_fns
                    .iter_mut()
                    .zip(&mut self.returns[agent.car_id as usize])
                    .zip(&self.combined_reward_weights)
                    .zip(&self.log_only_reward)
                    .map(|(((func, ret), weight), reward_bool)| {
                        // TODO: depreciate previous_act
                        // let agent_act = agent.last_actions;
                        // let previous_act = [
                        //     agent_act.throttle,
                        //     agent_act.steer,
                        //     agent_act.pitch,
                        //     agent_act.yaw,
                        //     agent_act.roll,
                        //     agent_act.jump as i32 as f32,
                        //     agent_act.boost as i32 as f32,
                        //     agent_act.handbrake as i32 as f32
                        // ];
                        let val = func.get_final_reward(agent, state);
                        let reward = val * *weight;
                        *ret += reward;
                        if *reward_bool {
                            reward
                        } else {
                            0.0
                        }
                    })
                    .sum();

                if agent.team_num == BLUE_TEAM {
                    self.blue_team_vals.insert(agent.car_id, final_val);
                } else {
                    self.orange_team_vals.insert(agent.car_id, final_val);
                }
            }

            // calculate team spirit rewards
            let blue_team_val_sum = self.blue_team_vals.values().sum::<f32>();
            let blue_team_val_mean = blue_team_val_sum / self.blue_team_vals.len() as f32;

            let orange_team_val_sum = self.orange_team_vals.values().sum::<f32>();
            let orange_team_val_mean = orange_team_val_sum / self.orange_team_vals.len() as f32;

            // let mut dbg_val = 0.;
            // blue team spirit calc
            for value in self.blue_team_vals.values_mut() {
                *value = ((1. - self.team_spirit) * *value)
                    + (self.team_spirit * blue_team_val_mean)
                    - orange_team_val_mean;
                // dbg_val += *value;
            }

            // orange team spirit calc
            for value in self.orange_team_vals.values_mut() {
                *value = ((1. - self.team_spirit) * *value)
                    + (self.team_spirit * orange_team_val_mean)
                    - blue_team_val_mean;
                // dbg_val += *value;
            }

            // assert!(dbg_val < 0.01, "dbg_val was too big, {dbg_val}, blue_team_vals: {:?}, orange_team_vals: {:?}", self.blue_team_vals, self.orange_team_vals);

            self.last_tick_calculated = state.tick_num;
        }

        let local_ret = self.returns[player.car_id as usize].clone();

        self.reward_sender.send(local_ret).unwrap();

        if player.team_num == BLUE_TEAM {
            self.blue_team_vals.get(&player.car_id).unwrap() * self.final_mult
        } else {
            self.orange_team_vals.get(&player.car_id).unwrap() * self.final_mult
        }
    }
}

/// "Wrapper" that collects a set of boxed reward functions and iterates through them to get a single float.
/// Has other functionality including reward logging that sends info to a separate singular thread which writes for all instances
/// instead of each instance having its own writer
// pub struct SB3CombinedLogRewardMultInst {
//     // reward_file_path: PathBuf,
//     // reward_file: String,
//     // lockfile: String,
//     reward_sender: Sender<Vec<f32>>,
//     final_mult: f32,
//     returns: Vec<Vec<f32>>,
//     combined_reward_fns: Vec<Box<dyn RewardFn + Send>>,
//     combined_reward_weights: Vec<f32>,
//     log_only_reward: Vec<bool>,
// }

// impl SB3CombinedLogRewardMultInst {
//     fn new(
//         reward_structs: Vec<Box<dyn RewardFn + Send>>,
//         reward_weights: Vec<f32>,
//         log_only_vec: Vec<bool>,
//         _file_location: Option<String>,
//         final_mult: Option<f32>,
//         sender: Sender<Vec<f32>>,
//     ) -> Self {
//         let final_mult = final_mult.unwrap_or(1.);

//         let returns = vec![vec![0.; reward_structs.len()]; 8];

//         SB3CombinedLogRewardMultInst {
//             reward_sender: sender,
//             final_mult,
//             returns,
//             combined_reward_fns: reward_structs,
//             combined_reward_weights: reward_weights,
//             log_only_reward: log_only_vec,
//         }
//     }
// }

// impl RewardFn for SB3CombinedLogRewardMultInst {
//     fn reset(&mut self, _initial_state: &GameState) {
//         for func in &mut self.combined_reward_fns {
//             func.reset(_initial_state);
//         }
//         for returns in &mut self.returns {
//             returns.fill(0.);
//         }
//     }

//     fn get_reward(&mut self, player: &PlayerData, state: &GameState, previous_action: &[f32]) -> f32 {
//         let final_val: f32 = self.combined_reward_fns
//             .iter_mut()
//             .zip(&mut self.returns[player.car_id as usize])
//             .zip(&self.combined_reward_weights)
//             .zip(&self.log_only_reward)
//             .map(|(((func, ret), weight), log_bool)| {
//                 let val = func.get_reward(player, state, previous_action);
//                 let reward = val * *weight;
//                 *ret += reward;
//                 if *log_bool {
//                     reward
//                 } else {
//                     0.0
//                 }
//             })
//             .sum();

//         final_val * self.final_mult
//     }

//     fn get_final_reward(&mut self, player: &PlayerData, state: &GameState, previous_action: &[f32]) -> f32 {
//         let final_val: f32 = self.combined_reward_fns
//             .iter_mut()
//             .zip(&mut self.returns[player.car_id as usize])
//             .zip(&self.combined_reward_weights)
//             .zip(&self.log_only_reward)
//             .map(|(((func, ret), weight), log_bool)| {
//                 let val = func.get_final_reward(player, state, previous_action);
//                 let reward = val * *weight;
//                 *ret += reward;
//                 if *log_bool {
//                     reward
//                 } else {
//                     0.0
//                 }
//             })
//             .sum();

//         let local_ret = self.returns[player.car_id as usize].clone();

//         // for returns in &mut self.returns {
//         //     returns.fill(0.);
//         // }
//         // self.returns[player.car_id as usize].fill(0.);

//         self.reward_sender.send(local_ret).unwrap();

//         final_val * self.final_mult
//     }
// }

/// "Wrapper" that collects a set of boxed reward functions and iterates through them to get a single float.
/// Has other functionality including reward logging. Unlike the mult-instance version, it writes on its own new thread.
pub struct SB3CombinedLogReward {
    reward_file_path: PathBuf,
    // reward_file: String,
    // lockfile: String,
    final_mult: f32,
    returns: Vec<f32>,
    combined_reward_fns: Vec<Box<dyn RewardFn>>,
    combined_reward_weights: Vec<f32>,
}

impl SB3CombinedLogReward {
    fn new(
        reward_structs: Vec<Box<dyn RewardFn>>,
        reward_weights: Vec<f32>,
        file_location: Option<String>,
        final_mult: Option<f32>,
    ) -> Self {
        let file_location = match file_location {
            Some(file_location) => file_location,
            None => "./combinedlogfiles".to_owned(),
        };

        let reward_file = format!("{}/rewards_test.txt", file_location);
        let reward_file_path = Path::new(&reward_file);
        // let lockfile = format!("{}/reward_lock", file_location);

        let final_mult = final_mult.unwrap_or(1.);
        let exists = Path::new(&file_location).exists();
        if !exists {
            let res = create_dir(&file_location);
            match res {
                Err(error) => {
                    if error.kind() == AlreadyExists {
                    } else {
                        panic!("{error}")
                    }
                }
                Ok(out) => out,
            }
        }
        for i in 0..100 {
            if i == 99 {
                panic!("too many attempts taken to lock the file in new")
            }

            let out = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(reward_file_path);

            let file = match out {
                Err(out) => {
                    if out.kind() == PermissionDenied {
                        continue;
                    } else {
                        println!("{out}");
                        continue;
                    }
                }
                Ok(_file) => _file,
            };

            let out = file.lock_exclusive();

            match out {
                // Err(out) => {if out.kind() == PermissionDenied {continue} else {continue}},
                Err(_) => continue,
                Ok(_file) => _file,
            };

            file.unlock().unwrap();
            break;
        }

        SB3CombinedLogReward {
            reward_file_path: reward_file_path.to_owned(),
            // reward_file: reward_file,
            // lockfile: lockfile,
            final_mult,
            returns: vec![0.; reward_structs.len()],
            combined_reward_fns: reward_structs,
            combined_reward_weights: reward_weights,
        }
    }
}

impl RewardFn for SB3CombinedLogReward {
    fn reset(&mut self, _initial_state: &GameState) {
        // self.returns = vec![0.; self.combined_reward_fns.len()];
        for func in &mut self.combined_reward_fns {
            func.reset(_initial_state);
        }
        self.returns.fill(0.);
    }

    fn get_reward(&mut self, player: &PlayerData, state: &GameState) -> f32 {
        let mut rewards = Vec::<f32>::new();

        for func in &mut self.combined_reward_fns {
            let val = func.get_reward(player, state);
            rewards.push(val);
        }

        let vals = element_mult_vec(&rewards, &self.combined_reward_weights);
        self.returns = element_add_vec(&self.returns, &vals);
        let sum = vals.iter().sum::<f32>();
        sum * self.final_mult
    }

    fn get_final_reward(&mut self, player: &PlayerData, state: &GameState) -> f32 {
        let mut rewards = Vec::<f32>::new();

        for func in &mut self.combined_reward_fns {
            let val = func.get_final_reward(player, state);
            rewards.push(val);
        }

        let vals = element_mult_vec(&rewards, &self.combined_reward_weights);
        // self.returns = element_add_vec(&self.returns, &vals);
        let local_ret = element_add_vec(&self.returns, &vals);

        // let local_ret = self.returns.clone();
        self.returns.fill(0.);
        let reward_file = self.reward_file_path.clone();

        thread::spawn(move || file_put(local_ret, reward_file.as_path()));

        let sum = vals.iter().sum::<f32>();
        sum * self.final_mult
    }
}

fn file_put(returns_local: Vec<f32>, reward_file: &Path) {
    for i in 0..100 {
        if i == 99 {
            panic!("too many attempts taken to lock the file in file_put")
        }
        let out = OpenOptions::new().append(true).read(true).open(reward_file);

        let file = match out {
            Err(out) => {
                println!("file error: {out}");
                // if out.kind() == PermissionDenied {continue} else {continue};},
                continue;
            }
            Ok(_file) => _file,
        };

        let out = file.lock_exclusive();

        match out {
            Err(out) => {
                println!("lock error: {out}");
                // if out.kind() == PermissionDenied {continue} else {continue};
                continue;
            }
            Ok(_file) => _file,
        };

        let mut buf = BufWriter::new(&file);

        let mut string = String::new();
        string += "[";
        for ret in returns_local.iter().take(returns_local.len() - 1) {
            string = string + &format!("{}, ", ret)
        }
        string = string + &format!("{}]", returns_local[returns_local.len() - 1]);
        writeln!(&mut buf, "{}", string).unwrap();
        let out = buf.flush();
        match out {
            Ok(out) => out,
            Err(err) => println!("buf.flush in logger failed with error: {err}"),
        };
        file.unlock().unwrap();
        break;
    }
}
