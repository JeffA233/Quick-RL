use std::{io::Cursor, time::Duration};

use indicatif::{MultiProgress, ProgressBar};
use redis::{Client, Commands};
use serde::{Deserialize, Serialize};
use tch::{nn, Device, Kind, Tensor};

use crate::{models::{model_base::DiscreteActPPO, ppo::default_ppo::{Actor, LayerConfig}}, vec_gym_env::VecGymEnv};

#[derive(Serialize, Deserialize)]
pub struct ExperienceStore {
    pub s_states: Vec<Vec<Vec<f32>>>,
    pub s_rewards: Vec<Vec<f32>>,
    pub s_actions: Vec<Vec<f32>>,
    pub dones_f: Vec<Vec<f32>>,
    pub s_log_probs: Vec<Vec<f32>>,
}


pub fn get_experience(
    nsteps: i64, 
    nprocs: i64, 
    obs_space: i64, 
    device: Device, 
    multi_prog_bar_total: &MultiProgress, 
    total_prog_bar: &ProgressBar, 
    prog_bar_func: impl Fn(u64) -> ProgressBar,
    // act_model: &mut Actor,
    // act_model_stream: Vec<u8>,
    redis_url: &str,
    act_model_config: LayerConfig,
    env: &mut VecGymEnv,
    sum_rewards: &mut Tensor,
    total_rewards: &mut f64,
    total_episodes: &mut f64,
) {
    // setup tensors for storage
    let s_states_ten = Tensor::zeros([nsteps + 1, nprocs, obs_space], (Kind::Float, device));
    // let mut s_states: Vec<Vec<Vec<f32>>> = Vec::with_capacity((nsteps + 1) as usize);
    // s_states.push(vec![vec![0.; obs_space as usize]; nprocs as usize]);
    let s_rewards_ten = Tensor::zeros([nsteps, nprocs], (Kind::Float, Device::Cpu));
    // let mut s_rewards: Vec<Vec<f32>> = Vec::with_capacity(nsteps as usize);
    let s_actions_ten = Tensor::zeros([nsteps, nprocs], (Kind::Int64, Device::Cpu));
    // let mut s_actions: Vec<Vec<f32>> = Vec::with_capacity(nsteps as usize);
    let dones_f_ten = Tensor::zeros([nsteps, nprocs], (Kind::Float, Device::Cpu));
    // let mut dones_f: Vec<Vec<f32>> = Vec::with_capacity(nsteps as usize);
    let s_log_probs_ten = Tensor::zeros([nsteps, nprocs], (Kind::Float, Device::Cpu));
    // let mut s_log_probs: Vec<Vec<f32>> = Vec::with_capacity(nsteps as usize);
    // progress bar
    let prog_bar = multi_prog_bar_total.add(prog_bar_func((nsteps * nprocs) as u64));
    prog_bar.set_message("getting rollouts");
    // take bytes from actor var store and load into new store (testing for networking later)
    let mut p = nn::VarStore::new(device);
    let mut act_model = Actor::new(&p.root(), act_model_config, None);
    // let env_observation_spc = env.observation_space();

    // get Redis connection
    let redis_client = Client::open(redis_url).unwrap();
    let mut redis_con = redis_client.get_connection_with_timeout(Duration::from_secs(30)).unwrap();
    let act_model_stream = redis_con.get::<&str, std::vec::Vec<u8>>("model_data").unwrap();

    let stream = Cursor::new(act_model_stream);
    p.load_from_stream(stream).unwrap();
    // let mut obs_store = Tensor::zeros([nprocs, obs_space], (Kind::Float, device));

    // redis_con.rpush::<&str, std::vec::Vec<Vec<f32>>, ()>("states", vec![vec![0.; obs_space as usize]; nprocs as usize]).unwrap();
    // start of experience gather loop
    for s in 0..nsteps {
        total_prog_bar.inc(nprocs as u64);
        prog_bar.inc(nprocs as u64);

        // let (actions, log_prob) = tch::no_grad(|| act_model.get_act_prob(&obs_store.to_device_(device, Kind::Float, true, false), false));
        let (actions, log_prob) = tch::no_grad(|| act_model.get_act_prob(&s_states_ten.get(s).to_device_(device, Kind::Float, true, false), false));
        // let probs = actor.softmax(-1, Kind::Float).view((-1, env.action_space()));
        // print_tensor_2df32("probs", &probs);
        // let actions = probs.clamp(1e-11, 1.).multinomial(1, true);
        // gather is used here to line up the probability with the action being done
        // let log_prob = probs.log().gather(-1, &actions, false);
        // print_tensor_2df32("acts", &actions);
        let actions_sqz = actions.squeeze().to_device_(Device::Cpu, Kind::Int64, true, false);
        // print_tensor_vecf32("acts flat", &actions_sqz);
        // print_tensor_2df32("log probs", &log_prob);
        let log_prob_flat = log_prob.squeeze().to_device_(Device::Cpu, Kind::Float, true, false);
        // print_tensor_vecf32("log prob flat", &log_prob_flat);
        let step = env.step(Vec::<i64>::try_from(&actions_sqz).unwrap(), Device::Cpu);

        *sum_rewards += &step.reward;
        // let ten_rew = Tensor::from_slice(&step.reward);
        // let ten_done = Tensor::from_slice(&step.is_done);
        *total_rewards +=
            f64::try_from((&*sum_rewards * &step.is_done).sum(Kind::Float)).unwrap();
        // let summed_rews = step.reward.iter().zip(step.is_done).map(|(rew, done)| rew * done as i16 as f32).sum::<f32>() as f64;
        // *total_rewards += step.reward.iter().zip(step.is_done.clone()).map(|(rew, done)| rew * (done as i16 as f32)).sum::<f32>() as f64;
        *total_episodes += f64::try_from(step.is_done.sum(Kind::Float)).unwrap();
        // let is_done_f = step.is_done.iter().map(|v| *v as i32 as f32).collect::<Vec<f32>>();
        // *total_episodes += is_done_f.iter().sum::<f32>() as f64;

        let masks = &step.is_done.to_kind(Kind::Float);
        // we want to flip so that we multiply by 0 every time is_done is set to true
        *sum_rewards *= &step.is_done.bitwise_not();
        // redis_con.rpush::<&str, std::vec::Vec<f32>, ()>("acts", Vec::try_from(actions_sqz).unwrap()).unwrap();
        // s_actions.push(Vec::try_from(actions_sqz).unwrap());
        // redis_con.rpush::<&str, std::vec::Vec<Vec<f32>>, ()>("states", step.obs.clone()).unwrap();
        // s_states.push(step.obs.clone());
        // redis_con.rpush::<&str, std::vec::Vec<f32>, ()>("log_probs", Vec::try_from(log_prob_flat).unwrap()).unwrap();
        // s_log_probs.push(Vec::try_from(log_prob_flat).unwrap());
        // redis_con.rpush::<&str, std::vec::Vec<f32>, ()>("rewards", step.reward).unwrap();
        // s_rewards.push(step.reward);
        // redis_con.rpush::<&str, std::vec::Vec<f32>, ()>("dones", is_done_f).unwrap();
        // dones_f.push(is_done_f);
        // s_states_ten.get(s + 1).copy_(&Tensor::from_slice2(&step.obs).view_(env_observation_spc.clone()).to_device_(Device::Cpu, Kind::Float, true, false),);
        // obs_store = Tensor::from_slice2(&step.obs).view_(&env_observation_spc).to_device_(device, Kind::Float, true, false);
        s_actions_ten.get(s).copy_(&actions_sqz);
        s_states_ten.get(s + 1).copy_(&step.obs);
        s_rewards_ten.get(s).copy_(&step.reward);
        dones_f_ten.get(s).copy_(masks);
        s_log_probs_ten.get(s).copy_(&log_prob_flat);
    }

    let s_states = Vec::try_from(s_states_ten).unwrap();
    let s_rewards = Vec::try_from(s_rewards_ten).unwrap();
    let s_actions = Vec::try_from(s_actions_ten).unwrap();
    let dones_f = Vec::try_from(dones_f_ten).unwrap();
    let s_log_probs = Vec::try_from(s_log_probs_ten).unwrap();

    let exp_store = ExperienceStore {s_states, s_rewards, s_actions, dones_f, s_log_probs};
    let mut s = flexbuffers::FlexbufferSerializer::new();
    exp_store.serialize(&mut s).unwrap();
    let view = s.view();
    // let exp_buf = s.take_buffer();
    redis_con.set::<&str, Vec<u8>, ()>("exp_store", view.to_vec()).unwrap();
    // redis_con.set::<&str, Vec<u8>, ()>("acts", act_buf).unwrap();
    // redis_con.set::<&str, Vec<Vec<Vec<f32>>>, ()>("states", s_states).unwrap();
    // redis_con.set::<&str, Vec<Vec<f32>>, ()>("log_probs", s_log_probs).unwrap();
    // redis_con.set::<&str, Vec<Vec<f32>>, ()>("rewards", s_rewards).unwrap();
    // redis_con.set::<&str, Vec<Vec<f32>>, ()>("dones", dones_f).unwrap();

    prog_bar.finish_and_clear();

    // (s_states, s_rewards, s_actions, dones_f, s_log_probs)
    // s_states_ten
}