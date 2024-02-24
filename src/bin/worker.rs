use std::{env, ffi::OsString, thread, time::Duration};

use crossbeam_channel::bounded;
/* Proximal Policy Optimization (PPO) model.

   Proximal Policy Optimization Algorithms, Schulman et al. 2017
   https://arxiv.org/abs/1707.06347

   See https://spinningup.openai.com/en/latest/algorithms/ppo.html for a
   reference python implementation.
*/
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use redis::{Client, Commands};
use serde::Deserialize;

use tch::Device;

use quick_rl::{
    algorithms::common_utils::{
        gather_experience::ppo_gather::get_experience, 
        rollout_buffer::rollout_buffer_worker::buffer_worker
    }, config::Configuration, models::ppo::default_ppo::LayerConfig, vec_gym_env::VecGymEnv
};

// NPROCS needs to be even to function properly (2 agents per 1v1 match)
// const MULT: i64 = 12;
// const MULT: i64 = 48;
// // const MULT: i64 = 1;
// const NPROCS: i64 = 2*MULT;
// // const NPROCS: i64 = 1;
// const NSTEPS: i64 = (2048*32)/NPROCS;
// // const NSTEPS: i64 = 1;
// const UPDATES: i64 = 1000000;


pub fn main() {
    // NOTE:
    // rough benchmark is ~4.26 for rew by idx 150
    // --- env setup stuff ---
    // I hate this path stuff but I'm not sure what's cleaner
    let mut config_path = env::current_exe().unwrap();
    config_path.pop();
    config_path.pop();
    config_path.pop();
    config_path.push("src/config.json");
    let config = match Configuration::load_configuration(config_path.as_path()) {
        Ok(config) => config,
        Err(error) => {
            panic!("Error loading configuration from '{}': {}", config_path.display(), error);
        }
    };
    let tick_skip = config.tick_skip;
    // let device = Device::Cpu;
    let device = Device::cuda_if_available();
    let reward_file_name = "./rewards_test.txt".to_owned();
    tch::manual_seed(0);
    tch::Cuda::manual_seed_all(0);

    let n_procs = config.n_env;
    let n_steps = config.hyperparameters.steps_per_rollout;
    let updates = config.hyperparameters.updates;
    // configure number of agents and gamemodes
    // let num_1s = (n_procs/2) as usize;
    // let num_1s_gravboost = 0;
    // let num_1s_selfplay = (n_procs/2) as usize;
    // let num_2s = 0;
    // let num_2s_gravboost = 0;
    // let num_2s_selfplay = 0;
    // // let num_3s = (n_procs/6) as usize;
    // let num_3s = 0;
    // let num_3s_gravboost = 0;
    // // let num_3s_selfplay = (n_procs/6) as usize;
    // let num_3s_selfplay = 0;
    // configure number of agents and gamemodes
    let num_1s = config.gamemodes.num_1s;
    let num_1s_gravboost = config.gamemodes.num_1s_gravboost;
    let num_1s_selfplay = config.gamemodes.num_1s_selfplay;
    let num_2s = config.gamemodes.num_2s;
    let num_2s_gravboost = config.gamemodes.num_2s_gravboost;
    let num_2s_selfplay = config.gamemodes.num_2s_selfplay;
    let num_3s = config.gamemodes.num_3s;
    let num_3s_gravboost = config.gamemodes.num_3s_gravboost;
    let num_3s_selfplay = config.gamemodes.num_3s_selfplay;

    let mut match_nums = Vec::new();
    match_nums.extend(vec![2; num_1s]);
    match_nums.extend(vec![4; num_2s]);
    match_nums.extend(vec![6; num_3s]);

    let mut gravity_nums = Vec::new();
    let mut boost_nums = Vec::new();
    for _ in 0..(num_1s - num_1s_gravboost) {
        gravity_nums.push(1.);
        boost_nums.push(1.);
    }
    for _ in 0..num_1s_gravboost {
        gravity_nums.push(0.);
        boost_nums.push(0.);
    }
    for _ in 0..(num_2s - num_2s_gravboost) {
        gravity_nums.push(1.);
        boost_nums.push(1.);
    }
    for _ in 0..num_2s_gravboost {
        gravity_nums.push(0.);
        boost_nums.push(0.);
    }
    for _ in 0..(num_3s - num_3s_gravboost) {
        gravity_nums.push(1.);
        boost_nums.push(1.);
    }
    for _ in 0..num_3s_gravboost {
        gravity_nums.push(0.);
        boost_nums.push(0.);
    }

    let mut self_plays = Vec::new();
    self_plays.extend(vec![false; num_1s - num_1s_selfplay]);
    self_plays.extend(vec![true; num_1s_selfplay]);
    self_plays.extend(vec![false; num_2s - num_2s_selfplay]);
    self_plays.extend(vec![true; num_2s_selfplay]);
    self_plays.extend(vec![false; num_3s - num_3s_selfplay]);
    self_plays.extend(vec![true; num_3s_selfplay]);

    // make progress bar
    let prog_bar_func = |len: u64| {
        let bar = ProgressBar::new(len);
        bar.set_style(ProgressStyle::with_template("[{per_sec}]").unwrap());
        bar
    };
    let multi_prog_bar_total = MultiProgress::new();
    let total_prog_bar = multi_prog_bar_total.add(prog_bar_func(updates as u64));

    // make env
    let mut env = VecGymEnv::new(match_nums, gravity_nums, boost_nums, self_plays, tick_skip, reward_file_name);
    println!("action space: {}", env.action_space());
    let obs_space = env.observation_space()[1];
    println!("observation space: {:?}", obs_space);

    // get Redis connection
    let db = config.redis.dbnum.clone();
    let password = if !config.redis.password_env_var.is_empty() {
        env::var_os(&config.redis.password_env_var).expect("Failed to get password") 
        }
        else{
            OsString::from("")
        };
    let password_str = password.to_str().expect("Failed to convert password to str");
    let redis_address = config.redis.ipaddress;
    let redis_str = format!("redis://{}:{}@{}/{}", config.redis.username, password_str, redis_address, db);
    let redis_str = redis_str.as_str();
    let redis_client = Client::open(redis_str).unwrap();
    let mut redis_con = redis_client.get_connection_with_timeout(Duration::from_secs(30)).unwrap();

    // moved here from gather_experience since otherwise we have to wait for full episodes to be submitted which is bad
    let (send_local, rx) = bounded(5000);
    let worker_url = redis_str.to_owned();
    let join_hand = thread::spawn(move || buffer_worker(rx, worker_url, obs_space, n_steps, n_procs as usize));

    let act_config_data = redis_con.get::<&str, Vec<u8>>("actor_structure").unwrap();
    let flex_read = flexbuffers::Reader::get_root(act_config_data.as_slice()).unwrap();
    let act_config = LayerConfig::deserialize(flex_read).unwrap();

    // misc stats stuff
    let mut sum_rewards = vec![0.; n_procs as usize];
    let mut total_rewards = 0f64;
    let mut total_episodes = 0f64;
    let mut total_steps = 0i64;

    // start of learning loops
    let mut update_index: i64 = 0;
    // for update_index in 0..UPDATES {
    loop {
        // TODO: in case we are also trying to learn and both are on GPU, we should pause the local worker to save resources,
        // though we also probably want to make this toggleable so that we can also run non-local workers
        let pause = redis_con.get::<&str, bool>("gather_pause").unwrap();

        if pause {
            thread::sleep(Duration::from_millis(100));
            continue;
        }

        get_experience(
            n_steps, 
            n_procs, 
            device, 
            &multi_prog_bar_total, 
            &total_prog_bar, 
            prog_bar_func, 
            redis_str,
            act_config.clone(),
            &mut env, 
            &send_local,
            &mut sum_rewards, 
            &mut total_rewards, 
            &mut total_episodes
        );

        total_steps += n_steps * n_procs;

        if update_index > 0 && update_index % 25 == 0 {
            println!("update idx: {}, total eps: {:.0}, episode rewards: {}, total steps: {}",
             update_index, total_episodes, total_rewards / total_episodes, total_steps);
            total_rewards = 0.;
            total_episodes = 0.;
        }
        update_index += 1;
    }
}