use std::{thread, time::Duration};

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
    }, 
    models::ppo::default_ppo::LayerConfig, 
    // tch_utils::dbg_funcs::{
    //     print_tensor_2df32, 
    //     print_tensor_noval, 
    //     print_tensor_vecf32
    // }
    vec_gym_env::VecGymEnv,
};

// NPROCS needs to be even to function properly (2 agents per 1v1 match)
// const MULT: i64 = 12;
const MULT: i64 = 48;
// const MULT: i64 = 1;
const NPROCS: i64 = 2*MULT;
// const NPROCS: i64 = 1;
const NSTEPS: i64 = (2048*32)/NPROCS;
// const NSTEPS: i64 = 1;
const UPDATES: i64 = 1000000;


pub fn main() {
    // NOTE:
    // rough benchmark is ~4.26 for rew by idx 150
    // --- env setup stuff ---
    let tick_skip = 8;
    // let device = Device::Cpu;
    let device = Device::cuda_if_available();
    let reward_file_name = "rewards_test".to_owned();
    tch::manual_seed(0);
    tch::Cuda::manual_seed_all(0);

    // configure number of agents and gamemodes
    let num_1s = (NPROCS/2) as usize;
    let num_1s_gravboost = 0;
    let num_1s_selfplay = (NPROCS/2) as usize;
    let num_2s = 0;
    let num_2s_gravboost = 0;
    let num_2s_selfplay = 0;
    // let num_3s = (NPROCS/6) as usize;
    let num_3s = 0;
    let num_3s_gravboost = 0;
    // let num_3s_selfplay = (NPROCS/6) as usize;
    let num_3s_selfplay = 0;

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
    let total_prog_bar = multi_prog_bar_total.add(prog_bar_func(UPDATES as u64));

    // make env
    let mut env = VecGymEnv::new(match_nums, gravity_nums, boost_nums, self_plays, tick_skip, reward_file_name);
    println!("action space: {}", env.action_space());
    let obs_space = env.observation_space()[1];
    println!("observation space: {:?}", obs_space);

    // get Redis connection
    let redis_str = "redis://127.0.0.1/";
    let redis_client = Client::open(redis_str).unwrap();
    let mut redis_con = redis_client.get_connection_with_timeout(Duration::from_secs(30)).unwrap();

    // moved here from gather_experience since otherwise we have to wait for full episodes to be submitted which is bad
    let (send_local, rx) = bounded(5000);
    let worker_url = redis_str.to_owned();
    let join_hand = thread::spawn(move || buffer_worker(rx, worker_url, obs_space, NSTEPS, NPROCS as usize));

    let act_config_data = redis_con.get::<&str, Vec<u8>>("actor_structure").unwrap();
    let flex_read = flexbuffers::Reader::get_root(act_config_data.as_slice()).unwrap();
    let act_config = LayerConfig::deserialize(flex_read).unwrap();

    // misc stats stuff
    let mut sum_rewards = vec![0.; NPROCS as usize];
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
            NSTEPS, 
            NPROCS, 
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

        total_steps += NSTEPS * NPROCS;

        if update_index > 0 && update_index % 25 == 0 {
            println!("update idx: {}, total eps: {:.0}, episode rewards: {}, total steps: {}",
             update_index, total_episodes, total_rewards / total_episodes, total_steps);
            total_rewards = 0.;
            total_episodes = 0.;
        }
        update_index += 1;
    }
}