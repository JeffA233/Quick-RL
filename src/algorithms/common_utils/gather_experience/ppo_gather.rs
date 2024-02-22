use std::{io::Cursor, thread, time::Duration};

use crossbeam_channel::bounded;
use indicatif::{MultiProgress, ProgressBar};
use redis::{Client, Commands};
// use serde::{Deserialize, Serialize};
use tch::{nn, Device, Kind, Tensor};

use crate::{
    algorithms::common_utils::rollout_buffer::rollout_buffer_worker::{buffer_worker, StepStore}, 
    models::{model_base::DiscreteActPPO, ppo::default_ppo::{Actor, LayerConfig}}, 
    vec_gym_env::VecGymEnv
};


pub fn get_experience(
    nsteps: i64, 
    nprocs: i64, 
    // obs_space: i64, 
    device: Device, 
    multi_prog_bar_total: &MultiProgress, 
    total_prog_bar: &ProgressBar, 
    prog_bar_func: impl Fn(u64) -> ProgressBar,
    redis_url: &str,
    act_model_config: LayerConfig,
    env: &mut VecGymEnv,
    // sum_rewards: &mut Tensor,
    sum_rewards: &mut [f64],
    total_rewards: &mut f64,
    total_episodes: &mut f64,
) {
    // setup tensors for storage
    let obs_space = env.observation_space()[1];
    // BUG: incorrect assumptions of dimensions
    // // let s_states_ten = Tensor::zeros([nsteps + 1, nprocs, obs_space], (Kind::Float, device));
    // let mut s_states: Vec<Vec<Vec<f32>>> = Vec::with_capacity((nsteps + 1) as usize);  // dimensions end up being [nsteps, nprocs, obs_space]
    // s_states.push(vec![vec![0.; obs_space as usize]; nprocs as usize]);
    // // let s_rewards_ten = Tensor::zeros([nsteps, nprocs], (Kind::Float, Device::Cpu));
    // let mut s_rewards: Vec<Vec<f32>> = Vec::with_capacity(nsteps as usize);
    // // let s_actions_ten = Tensor::zeros([nsteps, nprocs], (Kind::Int64, Device::Cpu));
    // let mut s_actions: Vec<Vec<f32>> = Vec::with_capacity(nsteps as usize);
    // // let dones_f_ten = Tensor::zeros([nsteps, nprocs], (Kind::Float, Device::Cpu));
    // let mut dones_f: Vec<Vec<f32>> = Vec::with_capacity(nsteps as usize);
    // // let s_log_probs_ten = Tensor::zeros([nsteps, nprocs], (Kind::Float, Device::Cpu));
    // let mut s_log_probs: Vec<Vec<f32>> = Vec::with_capacity(nsteps as usize);

    // let mut s_states: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(nsteps as usize); nprocs as usize];
    // for proc in s_states.iter_mut() {
    //     proc.push(vec![0.; obs_space as usize]);
    // }
    // let mut s_rewards: Vec<Vec<f32>> = vec![Vec::with_capacity(nsteps as usize); nprocs as usize];
    // let mut s_actions: Vec<Vec<f32>> = vec![Vec::with_capacity(nsteps as usize); nprocs as usize];
    // let mut dones_f: Vec<Vec<f32>> = vec![Vec::with_capacity(nsteps as usize); nprocs as usize];
    // let mut s_log_probs: Vec<Vec<f32>> = vec![Vec::with_capacity(nsteps as usize); nprocs as usize];
    // s_states.extend(vec![vec![0.; obs_space as usize]; nprocs as usize].iter());

    // progress bar
    let prog_bar = multi_prog_bar_total.add(prog_bar_func((nsteps * nprocs) as u64));
    prog_bar.set_message("getting rollouts");
    // setup actor model so we can later load the data (we only init the parameters here)
    let mut p = nn::VarStore::new(device);
    let mut act_model = Actor::new(&p.root(), act_model_config, None);

    let mut obs_store = Tensor::zeros([nprocs, obs_space], (Kind::Float, device));

    // get Redis connection
    let redis_client = Client::open(redis_url).unwrap();
    let mut redis_con = redis_client.get_connection_with_timeout(Duration::from_secs(30)).unwrap();

    // original non-threaded buffers
    // let mut rollout_bufs = Vec::new();
    // for _i in 0..nprocs as usize {
    //     rollout_bufs.push(RolloutBufferWorker::new(redis_url.to_owned(), obs_space, nsteps));
    // }

    // setup buffer threads
    // let mut recv_vec = Vec::<Receiver<StepStore>>::new();
    // let mut send_vec = Vec::<Sender<StepStore>>::new();
    // let mut thrd_vec = Vec::<JoinHandle<()>>::new();

    // for _i in 0..nprocs {
        let (send_local, rx) = bounded(5000);
        let worker_url = redis_url.to_owned();
        let join_hand = thread::spawn(move || buffer_worker(rx, worker_url, obs_space, nsteps, nprocs as usize));
        // thrd_vec.push(join_hand);
        // send_vec.push(send_local);
    // }

    let act_model_stream = redis_con.get::<&str, std::vec::Vec<u8>>("model_data").unwrap();
    let act_model_ver = redis_con.get::<&str, i64>("model_ver").unwrap();

    // load model bytes into VarStore (which then actually loads the parameters)
    let stream = Cursor::new(act_model_stream);
    p.load_from_stream(stream).unwrap();

    // start of experience gather loop
    // let mut all_done = false;
    let mut env_done_stores = vec![false; nprocs as usize];
    let mut s = 0;
    // for _s in 0..nsteps {
    while (s < nsteps) || !env_done_stores.iter().all(|b| *b) {
        total_prog_bar.inc(nprocs as u64);
        prog_bar.inc(nprocs as u64);

        // let (actions, log_prob) = tch::no_grad(|| act_model.get_act_prob(&s_states_ten.get(s).to_device_(device, Kind::Float, true, false), false));
        let (actions, log_prob) = tch::no_grad(|| act_model.get_act_prob(&obs_store, false));
        // print_tensor_2df32("probs", &probs);
        // print_tensor_2df32("acts", &actions);
        let actions_sqz = actions.squeeze().to_device_(Device::Cpu, Kind::Int64, true, false);
        // print_tensor_vecf32("acts flat", &actions_sqz);
        // print_tensor_2df32("log probs", &log_prob);
        let log_prob_flat = log_prob.squeeze().to_device_(Device::Cpu, Kind::Float, true, false);
        // print_tensor_vecf32("log prob flat", &log_prob_flat);
        let step = env.step(Vec::<i64>::try_from(&actions_sqz).unwrap(), Device::Cpu);

        // *sum_rewards += &step.reward;
        for (rew_sum, step_rew) in sum_rewards.iter_mut().zip(step.reward.iter()) {
            *rew_sum += *step_rew as f64;
        }

        // let ten_rew = Tensor::from_slice(&step.reward);
        // let ten_done = Tensor::from_slice(&step.is_done);
        // *total_rewards +=
        //     f64::try_from((&*sum_rewards * &step.is_done).sum(Kind::Float)).unwrap();
        *total_rewards += sum_rewards.iter().zip(step.is_done.iter()).map(|(r, b)| r * (*b as i32 as f64)).sum::<f64>();
        // let summed_rews = step.reward.iter().zip(step.is_done).map(|(rew, done)| rew * done as i16 as f32).sum::<f32>() as f64;
        // *total_rewards += step.reward.iter().zip(step.is_done.clone()).map(|(rew, done)| rew * (done as i16 as f32)).sum::<f32>() as f64;
        // *total_episodes += f64::try_from(step.is_done.sum(Kind::Float)).unwrap();
        let is_done_f = step.is_done.iter().map(|v| *v as i32 as f32).collect::<Vec<f32>>();
        *total_episodes += is_done_f.iter().sum::<f32>() as f64;

        // let masks = &step.is_done.to_kind(Kind::Float);

        // we want to flip so that we multiply by 0 every time is_done is set to true
        // *sum_rewards *= &step.is_done.bitwise_not();
        for (rew_sum, step_done) in sum_rewards.iter_mut().zip(step.is_done.iter()) {
            *rew_sum *= !*step_done as i32 as f64;
        }

        // s_states_ten.get(s + 1).copy_(&Tensor::from_slice2(&step.obs).view_(env_observation_spc.clone()).to_device_(Device::Cpu, Kind::Float, true, false),);
        // obs_store.copy_(&Tensor::from_slice2(&step.obs).view_([nprocs, obs_space]));
        obs_store = Tensor::from_slice2(&step.obs).to_device_(device, Kind::Float, true, false);

        let actions_vec = Vec::try_from(actions_sqz).unwrap();
        let log_probs_vec = Vec::try_from(log_prob_flat).unwrap();

        step.is_done.iter().zip(env_done_stores.iter_mut()).map(|(done, store)| if s > nsteps && *done {*store = *done}).for_each(drop);

        send_local.send(StepStore { obs: step.obs, action: actions_vec, reward: step.reward, done: is_done_f, log_prob: log_probs_vec, model_ver: act_model_ver }).unwrap();

        s += 1;
    }

    prog_bar.finish_and_clear();
}

// TODO: maybe convert function to struct so it can generate and hold the environment by itself, though we can do this with a function anyways so meh,
// also maybe look into dealing with the progress bar
// pub struct ExperienceGenerator {
//     nprocs: i64, 
//     // obs_space: i64, 
//     device: Device, 
//     // multi_prog_bar_total: &MultiProgress, 
//     // total_prog_bar: &ProgressBar, 
//     // prog_bar_func: impl Fn(u64) -> ProgressBar,
//     redis_url: String,
//     act_model_config: LayerConfig,
//     env: VecGymEnv,
//     sum_rewards: Tensor,
//     total_rewards: f64,
//     total_episodes: f64,
// }

// impl ExperienceGenerator {
//     // TODO: probably can calculate nprocs via the env_config
//     pub fn new(nprocs: i64, device: Device, redis_url: String, act_model_config: LayerConfig, env_config: EnvConfig) -> Self {
//         let env = VecGymEnv::new(env_config.match_nums, env_config.gravity_nums, env_config.boost_nums, env_config.self_plays, env_config.tick_skip, env_config.reward_file_name);
//         Self {
//             nprocs,
//             device,
//             redis_url,
//             act_model_config,
//             env,
//             sum_rewards: Tensor::zeros([nprocs], (Kind::Float, Device::Cpu)),
//             total_rewards: 0.,
//             total_episodes: 0.,
//         }
//     }
// }