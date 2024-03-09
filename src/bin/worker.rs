use std::{env, ffi::OsString, path::PathBuf, thread, time::Duration};

use crossbeam_channel::bounded;
/* Proximal Policy Optimization (PPO) model.

   Proximal Policy Optimization Algorithms, Schulman et al. 2017
   https://arxiv.org/abs/1707.06347

   See https://spinningup.openai.com/en/latest/algorithms/ppo.html for a
   reference python implementation.
*/
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
// use redis::{Client, Commands};
use serde::Deserialize;

use tch::Device;

use quick_rl::{
    algorithms::common_utils::{
        gather_experience::ppo_gather::get_experience, 
        rollout_buffer::rollout_buffer_worker::{buffer_worker, RedisWorkerBackend, RolloutWorkerBackend}
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
    // rough benchmark for reward is ~4.26 for rew (which is displayed only in the worker for now) by update idx 150 according to the learner side
    // this isn't quite reached with async for whatever reason(s)
    // --- env setup stuff ---
    // I hate this path stuff but I'm not sure what's cleaner
    let mut config_path = PathBuf::new();
    config_path.push("src/config.json");
    let config = match Configuration::load_configuration(config_path.as_path()) {
        Ok(config) => config,
        Err(error) => {
            panic!("Error loading configuration from '{}': {}", config_path.display(), error);
        }
    };

    let tick_skip = config.tick_skip;
    let device = Device::cuda_if_available();
    let reward_file_name = "./rewards_test.txt".to_owned();
    tch::manual_seed(0);
    tch::Cuda::manual_seed_all(0);

    let n_procs = config.n_env;
    let n_steps = config.hyperparameters.steps_per_rollout;
    let updates = config.hyperparameters.updates;

    // configure number of agents and gamemodes
    let mut team_size = Vec::new();
    team_size.extend(vec![1; config.gamemodes.num_1s]);
    team_size.extend(vec![2; config.gamemodes.num_2s]);
    team_size.extend(vec![3; config.gamemodes.num_3s]);

    let mut self_plays = Vec::new();
    self_plays.extend(vec![false; config.gamemodes.num_1s - config.gamemodes.num_1s_selfplay]);
    self_plays.extend(vec![true; config.gamemodes.num_1s_selfplay]);
    self_plays.extend(vec![false; config.gamemodes.num_2s - config.gamemodes.num_2s_selfplay]);
    self_plays.extend(vec![true; config.gamemodes.num_2s_selfplay]);
    self_plays.extend(vec![false; config.gamemodes.num_3s - config.gamemodes.num_3s_selfplay]);
    self_plays.extend(vec![true; config.gamemodes.num_3s_selfplay]);

    // make progress bar
    let prog_bar_func = |len: u64| {
        let bar = ProgressBar::new(len);
        bar.set_style(ProgressStyle::with_template("[{per_sec}]").unwrap());
        bar
    };
    let multi_prog_bar_total = MultiProgress::new();
    let total_prog_bar = multi_prog_bar_total.add(prog_bar_func(updates as u64));

    // make env
    let mut env = VecGymEnv::new(team_size, self_plays, tick_skip, reward_file_name);
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
    let mut backend = RedisWorkerBackend::new(redis_str.clone());
    // let backend_con = RolloutWorkerRedis::new()


    // moved here from gather_experience since otherwise we have to wait for full episodes to be submitted which is bad
    let (send_local, rx) = bounded(5000);
    let worker_url = redis_str.to_owned();
    println!("we're about to spawn the thread");
    thread::spawn(move || {
        buffer_worker(rx, move || RedisWorkerBackend::new(worker_url.clone()), obs_space, n_steps, n_procs as usize);
    });
    
    let act_config_data = backend.get_key_value_raw("actor_structure").unwrap();
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
        let pause = backend.get_key_value_bool("gather_pause").unwrap();

        if pause {
            thread::sleep(Duration::from_millis(100));
            continue;
        }

        get_experience(
            &mut backend,
            n_steps, 
            n_procs, 
            device, 
            &multi_prog_bar_total, 
            &total_prog_bar, 
            prog_bar_func, 
            act_config.clone(),
            &mut env, 
            &send_local,
            &mut sum_rewards, 
            &mut total_rewards, 
            &mut total_episodes,
            config.network.act_func.clone(),
        );

        total_steps += n_steps * n_procs;

        if update_index > 0 && update_index % 25 == 0 {
            println!("worker loop idx: {}, total eps: {:.0}, episode rewards: {}, total steps: {}",
             update_index, total_episodes, total_rewards / total_episodes, total_steps);
            total_rewards = 0.;
            total_episodes = 0.;
        }
        update_index += 1;
    }
}