use rlgym_sim_gym::common_values::{BLUE_GOAL_CENTER, ORANGE_GOAL_CENTER};
use rlgym_sim_gym::gamestates::physics_object::Position;

use rlgym_sim_gym::DefaultState;
use rlgym_sim_gym::StateSetter;
use rand::distributions::weighted::WeightedIndex;
use rand::prelude::Distribution;
use rand::Rng;
use rand::{rngs::SmallRng, thread_rng, SeedableRng};
use rlgym_sim_gym::state_setters::wrappers::state_wrapper::StateWrapper;

use serde_json::from_reader;
use std::{fs::File, io::BufReader};
use zip::read::ZipArchive;

/// return the configured state setters for use with Matrix
pub fn custom_state_setters(team_size: usize, seed: Option<u64>) -> WeightedSampleSetter {
    // let replay_setter_str = if team_size == 1 {
    //     "replay_folder/ssl_1v1.zip".to_owned()
    // } else if team_size == 2 {
    //     "replay_folder/ssl_2v2.zip".to_owned()
    // } else {
    //     "replay_folder/ssl_3v3.zip".to_owned()
    // };

    // NOTE: we want to use the default state here so as to try to force consistency for the reward performance testing.
    let state_setters: Vec<Box<dyn StateSetter + Send>> = vec![
        Box::new(DefaultState::new(seed)),
        // Box::new(RandomState::new(None, None, Some(false), seed)),
        // Box::new(ReplaySetter::new(replay_setter_str)),
    ];
    WeightedSampleSetter::new(
        state_setters,
        // vec![0.3, 0.3, 0.15],
        vec![1.0],
        seed,
    )
}

/// weighted state setter that uses a rand distribution to poll for a choice
pub struct WeightedSampleSetter {
    state_setters: Vec<Box<dyn StateSetter + Send>>,
    distribution: WeightedIndex<f64>,
    rng: SmallRng,
}

impl WeightedSampleSetter {
    pub fn new(
        state_setters: Vec<Box<dyn StateSetter + Send>>,
        weights: Vec<f64>,
        seed: Option<u64>,
    ) -> Self {
        assert!(
            state_setters.len() == weights.len(),
            "WeightedSampleSetter requires the argument lengths match"
        );
        let distribution = WeightedIndex::new(&weights).unwrap();
        let seed = match seed {
            Some(seed) => seed,
            None => thread_rng().gen_range(0..10000),
        };
        let rng = SmallRng::seed_from_u64(seed);
        WeightedSampleSetter {
            state_setters,
            distribution,
            rng,
        }
    }
}

impl StateSetter for WeightedSampleSetter {
    fn reset(&mut self, state_wrapper: &mut StateWrapper) {
        let choice = self.distribution.sample(&mut self.rng);
        self.state_setters[choice].reset(state_wrapper);
    }

    fn set_seed(&mut self, seed: u64) {
        self.rng = SmallRng::seed_from_u64(seed);
        for state_setter in &mut self.state_setters {
            state_setter.set_seed(seed);
        }
    }
}

/// uses a cached (loaded from disk) set of replay states to later set the bot states to
pub struct ReplaySetter {
    states: Vec<Vec<f32>>,
    rng: SmallRng,
    margin_distance: f32,
}

impl ReplaySetter {
    pub fn new(file_str: String) -> Self {
        let file = File::open(file_str).unwrap();
        let mut zip_file = ZipArchive::new(file).unwrap();
        let zip_file = zip_file.by_index(0).unwrap();
        let reader = BufReader::new(zip_file);
        let states: Result<Vec<Vec<f32>>, serde_json::Error> = from_reader(reader);
        let states = match states {
            Ok(values) => values,
            Err(e) => panic!("err from replay setter when loading json: {e}"),
        };

        let seed = thread_rng().gen_range(0..10000);
        // let rng = StdRng::seed_from_u64(seed);
        let rng = SmallRng::seed_from_u64(seed);
        ReplaySetter {
            states,
            rng,
            margin_distance: 500.,
        }
    }

    fn set_cars(state_wrapper: &mut StateWrapper, state: &[f32]) {
        let data = &state[9..state_wrapper.cars.len() * 13 + 9];
        let mut i = 0;
        for car in state_wrapper.cars.iter_mut() {
            car.set_pos(Some(data[i]), Some(data[i + 1]), Some(data[i + 2]));
            car.set_rot(Some(data[i + 3]), Some(data[i + 4]), Some(data[i + 5]));
            car.set_lin_vel(Some(data[i + 6]), Some(data[i + 7]), Some(data[i + 8]));
            car.set_ang_vel(Some(data[i + 9]), Some(data[i + 10]), Some(data[i + 11]));
            car.boost = data[i + 12];
            i += 13;
        }
    }

    fn check_bounds(&self, ball_pos: Position) -> bool {
        let dist_blue: f32 = (ball_pos - BLUE_GOAL_CENTER)
            .into_iter()
            .map(|val| val.abs())
            .sum();
        if dist_blue < self.margin_distance {
            return true;
        }

        let dist_orange: f32 = (ball_pos - ORANGE_GOAL_CENTER)
            .into_iter()
            .map(|val| val.abs())
            .sum();
        if dist_orange < self.margin_distance {
            return true;
        }

        false
    }

    fn set_ball(state_wrapper: &mut StateWrapper, data: &[f32]) {
        state_wrapper
            .ball
            .set_pos(Some(data[0]), Some(data[1]), Some(data[2]));
        state_wrapper
            .ball
            .set_lin_vel(Some(data[3]), Some(data[4]), Some(data[5]));
        state_wrapper
            .ball
            .set_ang_vel(Some(data[6]), Some(data[7]), Some(data[8]));
    }
}

impl StateSetter for ReplaySetter {
    fn reset(&mut self, state_wrapper: &mut StateWrapper) {
        let mut i = 0;
        loop {
            if self.states.is_empty() {
                panic!("ran out of states in ReplaySetter")
            }
            let idx = self.rng.gen_range(0..self.states.len());
            let state = self.states[idx].clone();
            ReplaySetter::set_ball(state_wrapper, &state);
            // make sure the ball isn't too close to the net to make sure we don't have a quick match
            if self.check_bounds(state_wrapper.ball.position) {
                i += 1;
                self.states.remove(idx);
                if i != 100 {
                    continue;
                }
            }
            ReplaySetter::set_cars(state_wrapper, &state);
            break;
        }
        // let state = self.states[self.rng.gen_range(0..self.states.len())].clone();
        // ReplaySetter::_set_ball(state_wrapper, &state);
        // ReplaySetter::_set_cars(state_wrapper, &state);
    }
}

// in the future, for bigger replays so that we don't need to store them completely in memory
// pub struct ReplaySetter2 {
//     file_str: String,
//     rng: StdRng,
// }

// impl ReplaySetter2 {
//     pub fn new(file_str: String) -> Self {
//         let seed = thread_rng().gen_range(0..10000);
//         let rng = StdRng::seed_from_u64(seed);

//         Self { file_str, rng }
//     }

//     fn _set_cars(state_wrapper: &mut StateWrapper, state: &[f32]) {
//         let data = &state[9..state_wrapper.cars.len() * 13 + 9];
//         let mut i = 0;
//         for car in state_wrapper.cars.iter_mut() {
//             car.set_pos(Some(data[i]), Some(data[i + 1]), Some(data[i + 2]));
//             car.set_rot(Some(data[i + 3]), Some(data[i + 4]), Some(data[i + 5]));
//             car.set_lin_vel(Some(data[i + 6]), Some(data[i + 7]), Some(data[i + 8]));
//             car.set_ang_vel(Some(data[i + 9]), Some(data[i + 10]), Some(data[i + 11]));
//             car.boost = data[i + 12];
//             i += 13;
//         }
//     }

//     fn _set_ball(state_wrapper: &mut StateWrapper, data: &[f32]) {
//         state_wrapper.ball.set_pos(Some(data[0]), Some(data[1]), Some(data[2]));
//         state_wrapper.ball.set_lin_vel(Some(data[3]), Some(data[4]), Some(data[5]));
//         state_wrapper.ball.set_ang_vel(Some(data[6]), Some(data[7]), Some(data[8]));
//     }
// }

// impl StateSetter for ReplaySetter2 {
//     fn reset(&mut self, state_wrapper: &mut StateWrapper) {
//         let file = File::open(&self.file_str).unwrap();
//         let mut zip_file = ZipArchive::new(file).unwrap();
//         let file_index = self.rng.gen_range(0..zip_file.len());
//         let zip_file = zip_file.by_index(file_index).unwrap();
//         let reader = BufReader::new(zip_file);
//         let state_op: Result<Vec<f32>, serde_json::Error> = from_reader(reader);
//         let state = match state_op {
//             Ok(values) => values,
//             Err(e) => panic!("err from replay setter when loading json: {e}"),
//         };
//         Self::_set_ball(state_wrapper, &state);
//         Self::_set_cars(state_wrapper, &state);
//     }
// }
