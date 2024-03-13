use std::{
    io::Cursor,
    // thread,
    // time::Duration,
};

use crossbeam_channel::Sender;
use indicatif::{MultiProgress, ProgressBar};
// use redis::{Client, Commands};
// use serde::{Deserialize, Serialize};
use tch::{nn, Device, Kind, Tensor, no_grad_guard};

use crate::{
    algorithms::common_utils::rollout_buffer::{
        rollout_buffer_worker::{
            // RolloutWorkerBackend, 
            StepStore,
        },
        rollout_buffer_utils::RolloutDatabaseBackend,
    },
    
    models::{
        model_base::DiscreteActPPO,
        ppo::default_ppo::{Actor, LayerConfig},
    },
    vec_gym_env::VecGymEnv,
};

pub fn get_experience<T: RolloutDatabaseBackend>(
    backend: &mut T,
    nsteps: i64,
    nprocs: i64,
    device: Device,
    multi_prog_bar_total: &MultiProgress,
    total_prog_bar: &ProgressBar,
    prog_bar_func: impl Fn(u64) -> ProgressBar,
    act_model_config: LayerConfig,
    env: &mut VecGymEnv,
    send_local: &Sender<StepStore>,
    sum_rewards: &mut [f64],
    total_rewards: &mut f64,
    total_episodes: &mut f64,
    act_func: String,
) {
    let obs_space = env.observation_space()[1];

    // progress bar
    let prog_bar = multi_prog_bar_total.add(prog_bar_func((nsteps * nprocs) as u64));
    prog_bar.set_message("getting rollouts");
    // setup actor model so we can later load the data (we only init the parameters here)
    let mut p = nn::VarStore::new(device);
    let mut act_model = Actor::new(&p.root(), act_model_config, None, act_func);

    let mut obs_store = Tensor::zeros([nprocs, obs_space], (Kind::Float, device));

    // BUG: worker occasionally dies here
    let act_model_stream = backend.get_key_value_raw("model_data").unwrap();
    let act_model_ver = backend.get_key_value_i64("model_ver").unwrap();

    // load model bytes into VarStore (which then actually loads the parameters)
    let stream = Cursor::new(act_model_stream);
    p.load_from_stream(stream).unwrap();

    // start of experience gather loop
    let mut s = 0;
    let guard = no_grad_guard();
    while s < nsteps {
        total_prog_bar.inc(nprocs as u64);
        prog_bar.inc(nprocs as u64);

        // let (actions, log_prob) = tch::no_grad(|| act_model.get_act_prob(&obs_store, false));
        let (actions, log_prob) = act_model.get_act_prob(&obs_store, false);
        let actions_sqz = actions
            .squeeze()
            .to_device_(Device::Cpu, Kind::Int16, true, false);
        let log_prob_flat = log_prob
            .squeeze()
            .to_device_(Device::Cpu, Kind::Float, true, false);

        let step = env.step(Vec::<i64>::try_from(&actions_sqz).unwrap(), Device::Cpu);

        for (rew_sum, step_rew) in sum_rewards.iter_mut().zip(step.reward.iter()) {
            *rew_sum += *step_rew as f64;
        }

        *total_rewards += sum_rewards
            .iter()
            .zip(step.is_done.iter())
            .map(|(r, b)| r * (*b as i32 as f64))
            .sum::<f64>();

        let is_done_f = step
            .is_done
            .iter()
            .map(|v| *v as i32 as f32)
            .collect::<Vec<f32>>();
        *total_episodes += is_done_f.iter().sum::<f32>() as f64;

        // we want to flip so that we multiply by 0 every time is_done is set to true
        for (rew_sum, step_done) in sum_rewards.iter_mut().zip(step.is_done.iter()) {
            *rew_sum *= !*step_done as i32 as f64;
        }

        // s_states_ten.get(s + 1).copy_(&Tensor::from_slice2(&step.obs).view_(env_observation_spc.clone()).to_device_(Device::Cpu, Kind::Float, true, false),);
        // obs_store.copy_(&Tensor::from_slice2(&step.obs).view_([nprocs, obs_space]));
        obs_store = Tensor::from_slice2(&step.obs).to_device_(device, Kind::Float, true, false);

        let actions_vec = Vec::try_from(actions_sqz).unwrap();
        let log_probs_vec = Vec::try_from(log_prob_flat).unwrap();

        send_local
            .send(StepStore {
                obs: step.obs,
                action: actions_vec,
                reward: step.reward,
                done: is_done_f,
                log_prob: log_probs_vec,
                model_ver: act_model_ver,
            })
            .unwrap();

        s += 1;
    }

    prog_bar.finish_and_clear();
    drop(guard)
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
