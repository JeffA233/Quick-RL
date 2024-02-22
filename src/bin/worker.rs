use std::time::Duration;

/* Proximal Policy Optimization (PPO) model.

   Proximal Policy Optimization Algorithms, Schulman et al. 2017
   https://arxiv.org/abs/1707.06347

   See https://spinningup.openai.com/en/latest/algorithms/ppo.html for a
   reference python implementation.
*/
// use bytebuffer::ByteBuffer;
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
// use serde::{
//     // de::IntoDeserializer, 
//     Deserialize, 
//     // Deserializer
// };
// use tch::nn::init::{NonLinearity, NormalOrUniform};
// use quick_rl::vec_gym_env::VecGymEnv;
// use tch::kind::{FLOAT_CPU, INT64_CPU};
use tch::{nn::{self, init, LinearConfig, OptimizerConfig}, Device, Kind, Tensor};

use quick_rl::{
    algorithms::common_utils::{gather_experience::ppo_gather::get_experience, rollout_buffer::{
        rollout_buffer_host::RolloutBufferHost, 
        // rollout_buffer_utils::ExperienceStoreProcs
    }}, 
    models::{model_base::{DiscreteActPPO, Model}, ppo::default_ppo::{Actor, Critic, LayerConfig}}, 
    // tch_utils::dbg_funcs::{
    //     print_tensor_2df32, 
    //     print_tensor_noval, 
    //     print_tensor_vecf32
    // }
    vec_gym_env::VecGymEnv,
};

// use redis::{
//     Client, Commands,
// };

// const ENV_NAME: &str = "SpaceInvadersNoFrameskip-v4";
// NPROCS needs to be even to function properly (2 agents per 1v1 match)
// const MULT: i64 = 12;
const MULT: i64 = 48;
// const MULT: i64 = 1;
const NPROCS: i64 = 2*MULT;
// const NPROCS: i64 = 1;
const NSTEPS: i64 = (2048*32)/NPROCS;
// const NSTEPS: i64 = 6;
const NSTACK: i64 = 1;
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
    // let redis_client = Client::open(redis_str).unwrap();
    // let mut redis_con = redis_client.get_connection_with_timeout(Duration::from_secs(30)).unwrap();

    // let mut buffer_host = RolloutBufferHost::new(redis_str.to_owned());

    // setup actor and critic
    // let vs_act = nn::VarStore::new(device);
    // let vs_critic = nn::VarStore::new(device);
    // let init_config = Some(LinearConfig { ws_init: init::Init::Orthogonal { gain: 2_f64.sqrt() }, bs_init: Some(init::Init::Const(0.)), bias: true });
    // TODO: get this from Redis which then is from the learner
    let act_config = LayerConfig::new(vec![256; 3], obs_space, Some(env.action_space()));
    // let mut act_model = Actor::new(&vs_act.root(), act_config.clone(), init_config);
    // let critic_config = LayerConfig::new(vec![256; 3], obs_space, None);
    // let mut critic_model = Critic::new(&vs_critic.root(), critic_config, init_config);
    // let mut opt_act = nn::Adam::default().build(&vs_act, lr).unwrap();
    // let mut opt_critic = nn::Adam::default().build(&vs_critic, lr).unwrap();

    // redis_con.set::<&str, i64, ()>("model_ver", 0).unwrap();
    // let mut model_ver: i64 = 0;

    // misc stats stuff
    // let mut sum_rewards = Tensor::zeros([NPROCS], (Kind::Float, Device::Cpu));
    let mut sum_rewards = vec![0.; NPROCS as usize];
    let mut total_rewards = 0f64;
    let mut total_episodes = 0f64;
    let mut total_steps = 0i64;

    // let train_size = NSTEPS * NPROCS;
    // start of learning loops
    for update_index in 0..UPDATES {
        // let mut act_save_stream = ByteBuffer::new();
        // // TODO: consider parsing this result
        // vs_act.save_to_stream(&mut act_save_stream).unwrap();
        // let act_buffer = act_save_stream.into_vec();

        // redis_con.incr::<&str, i64, ()>("model_ver", 1).unwrap();
        // model_ver += 1;

        // redis_con.set::<&str, std::vec::Vec<u8>, ()>("model_data", act_buffer).unwrap();
        // // clear redis
        // redis_con.del::<&str, ()>("exp_store").unwrap();

        get_experience(
            NSTEPS, 
            NPROCS, 
            // obs_space, 
            device, 
            &multi_prog_bar_total, 
            &total_prog_bar, 
            prog_bar_func, 
            // &mut act_model, 
            // act_buffer,
            redis_str,
            act_config.clone(),
            &mut env, 
            &mut sum_rewards, 
            &mut total_rewards, 
            &mut total_episodes
        );
        // let ten_vec: Vec<Vec<Vec<f32>>> = Vec::try_from(ten_obs).unwrap();
        // -- 
        total_steps += NSTEPS * NPROCS;
        // get redis
        // let exp_store = redis_con.get::<&str, Vec<u8>>("exp_store").unwrap();
        // let flex_read = flexbuffers::Reader::get_root(exp_store.as_slice()).unwrap();

        // let exp_store = ExperienceStoreProcs::deserialize(flex_read).unwrap();
        // let mut exp_store = buffer_host.get_experience(BUFFERSIZE as usize, model_ver-min_model_ver);
        // exp_store.s_states.push(exp_store.terminal_obs);
        // // truncate since get_experience returns full episodes for now
        // // exp_store.s_states.truncate((NSTEPS * NPROCS) as usize);
        // // exp_store.s_actions.truncate((NSTEPS * NPROCS) as usize);
        // // exp_store.s_rewards.truncate((NSTEPS * NPROCS) as usize);
        // // exp_store.dones_f.truncate((NSTEPS * NPROCS) as usize);
        // // exp_store.s_log_probs.truncate((NSTEPS * NPROCS) as usize);
        // // println!("states size was: {}", exp_store.s_states.len());
        // // println!("states size 2 was: {}", exp_store.s_states[0].len());
        // // println!("states size 3 was: {}", exp_store.s_states[0][0].len());
        // //
        // // move to Tensor
        // let s_actions = Tensor::from_slice(&exp_store.s_actions);
        // let s_states = Tensor::from_slice2(&exp_store.s_states);
        // // let s_states = Tensor::zeros([NSTEPS + 1, NPROCS, obs_space], (Kind::Float, device));
        // // for (i, state) in exp_store.s_states.iter().enumerate() {
        // //     s_states.get(i as i64).copy_(&Tensor::from_slice2(state));
        // // }
        // // let ten_vec: Vec<Vec<Vec<f32>>> = Vec::try_from(&s_states).unwrap();
        // // println!("states size was: {}", ten_vec.len());
        // // println!("states size 2 was: {}", ten_vec[0].len());
        // // println!("states size 3 was: {}", ten_vec[0][0].len());
        // let s_log_probs = Tensor::from_slice(&exp_store.s_log_probs);
        // let s_rewards = Tensor::from_slice(&exp_store.s_rewards);
        // let dones_f = Tensor::from_slice(&exp_store.dones_f);
        // //
        // // print_tensor_noval("states", &s_states);  // size = [1025, 16, 107]
        // // print_tensor_noval("rewards", &s_rewards);
        // // print_tensor_noval("actions", &s_actions);
        // // print_tensor_noval("dones", &dones_f);
        // // print_tensor_noval("log_probs", &s_log_probs);
        // // TODO: NSTACK is probably not required here?
        // // let states = s_states.view([NSTEPS + 1, NPROCS, NSTACK, obs_space]);
        // // let states = s_states.view([NSTEPS * NPROCS, NSTACK, obs_space]);
        // let states = s_states;
        // let buf_size = s_rewards.size()[0];
        // // print_tensor_noval("states after view", &states);

        // // compute gae
        // // let adv = Tensor::zeros([NSTEPS, NPROCS], (Kind::Float, Device::Cpu));
        // let adv = Tensor::zeros([buf_size], (Kind::Float, Device::Cpu));
        // let vals = tch::no_grad(|| critic_model.forward(&states.to_device_(device, Kind::Float, true, false))).squeeze().to_device_(Device::Cpu, Kind::Float, true, false);
        // // print_tensor_noval("vals from critic", &vals);
        // // let mut last_gae_lam = Tensor::zeros([NPROCS], (Kind::Float, Device::Cpu));
        // let mut last_gae_lam = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        // for idx in (0..buf_size).rev() {
        //     let done = if idx == buf_size - 1 {
        //         1. - dones_f.get(idx)
        //     } else {
        //         1. - dones_f.get(idx + 1)
        //     };
        //     let next_val = vals.get(idx + 1);
        //     // print_tensor_f32("val", &val_idx);
        //     let rew = s_rewards.get(idx);
        //     // print_tensor_f32("rew", &rew);
        //     // print_tensor_noval("rew", &rew);
        //     // print_tensor_noval("next_val", &next_val);
        //     // print_tensor_noval("done", &done);
        //     let pred_ret = rew + gamma * &next_val * &done;
        //     // print_tensor_f32("pred_ret", &pred_ret);
        //     let delta = &pred_ret - &vals.get(idx);
        //     // print_tensor_f32("delta", &delta);
        //     last_gae_lam = delta + gamma * 0.95 * done * last_gae_lam;
        //     // print_tensor_f32("last_gae_lam", &last_gae_lam);
        //     adv.get(idx).copy_(&last_gae_lam.squeeze());
        //     // print_tensor_f32("targ_val", &targ_val);
        // }
        // // shrink everything to fit batch size
        // let advantages = adv.narrow(0, 0, NSTEPS * NPROCS).view([train_size, 1]);
        // // we now must do view on everything we want to batch
        // // let advantages = adv.view([train_size, 1]);
        // let target_vals = (&advantages + vals.narrow(0, 0, NSTEPS * NPROCS).view([train_size, 1])).to_device_(device, Kind::Float, true, false);

        // let learn_states = states.narrow(0, 0, NSTEPS * NPROCS).view([train_size, NSTACK, obs_space]).to_device_(device, Kind::Float, true, false);

        // // norm advantages
        // let advantages = ((&advantages - &advantages.mean(Kind::Float)) / (&advantages.std(true) + 1e-8)).to_device_(device, Kind::Float, true, false);
        
        // let actions = s_actions.narrow(0, 0, NSTEPS * NPROCS).view([train_size]).to_device_(device, Kind::Int64, true, false);
        // let old_log_probs = s_log_probs.narrow(0, 0, NSTEPS * NPROCS).view([train_size]).to_device_(device, Kind::Float, true, false);

        // let prog_bar = multi_prog_bar_total.add(prog_bar_func((OPTIM_EPOCHS) as u64));
        // prog_bar.set_message("doing epochs");
        // // stats
        // let mut clip_fracs = Vec::new();
        // let mut kl_divs = Vec::new();
        // let mut entropys = Vec::new();
        // let mut losses = Vec::new();
        // let mut act_loss = Vec::new();
        // let mut val_loss = Vec::new();

        // let optim_indexes = Tensor::randint(train_size, [OPTIM_EPOCHS, BUFFERSIZE], (Kind::Int64, device));
        // // learner epoch loop
        // for epoch in 0..OPTIM_EPOCHS {
        //     prog_bar.inc(1);
        //     let batch_indexes = optim_indexes.get(epoch);
        //     for batch_start_index in (0..BUFFERSIZE).step_by(OPTIM_BATCHSIZE as usize) {
        //         let buffer_indexes = batch_indexes.slice(0, batch_start_index, batch_start_index + OPTIM_BATCHSIZE, 1);
        //         let states = learn_states.index_select(0, &buffer_indexes);
        //         let actions = actions.index_select(0, &buffer_indexes);
        //         // print_tensor_vecf32("batch actions", &actions);
        //         let advs = advantages.index_select(0, &buffer_indexes).squeeze();
        //         // print_tensor_vecf32("batch advantages", &advs);
        //         let targ_vals = target_vals.index_select(0, &buffer_indexes).squeeze();
        //         // print_tensor_vecf32("batch targ vals", &targ_vals);
        //         let old_log_probs_batch = old_log_probs.index_select(0, &buffer_indexes).squeeze();
        //         // print_tensor_vecf32("batch old log probs", &old_log_probs_batch);
        //         let (action_log_probs, dist_entropy) = act_model.get_prob_entr(&states, &actions);
        //         let vals = critic_model.forward(&states).squeeze();
        //         // // print_tensor_vecf32("batch vals", &vals);
        //         let dist_entropy_float = tch::no_grad(|| {f32::try_from(&dist_entropy.detach()).unwrap()});
        //         entropys.push(dist_entropy_float);
    
        //         // PPO ratio
        //         let ratio = (&action_log_probs - &old_log_probs_batch).exp().squeeze();
        //         // print_tensor_vecf32("ratio", &ratio);
        //         let clip_ratio = ratio.clamp(1.0 - clip_range, 1.0 + clip_range);
        //         // print_tensor_vecf32("clip ratio", &clip_ratio);
        //         clip_fracs.push(tch::no_grad(|| {
        //             let est = ((&ratio - 1.).abs().greater(clip_range).to_kind(Kind::Float)).mean(Kind::Float);
        //             f32::try_from(&est.detach().to(Device::Cpu)).unwrap()
        //         }));
        //         kl_divs.push(tch::no_grad(|| {
        //             let log_ratio = &action_log_probs - &old_log_probs_batch;
        //             // let act_log_prob_mean = f64::try_from(&action_log_probs.mean(Kind::Float)).unwrap();
        //             // let old_log_probs_batch_mean = f64::try_from(&old_log_probs_batch.mean(Kind::Float)).unwrap();
        //             // let log_ratio_mean = f64::try_from(&log_ratio.mean(Kind::Float).detach()).unwrap();
        //             let kl = (log_ratio.exp() - 1.) - log_ratio;
        //             // for dbg
        //             // act_log_prob_mean;
        //             // old_log_probs_batch_mean;
        //             // log_ratio_mean;
        //             f32::try_from(kl.mean(Kind::Float).detach().to(Device::Cpu)).unwrap()
        //         }));
    
        //         let value_loss = &vals.mse_loss(&targ_vals.squeeze(), tch::Reduction::Mean).squeeze();
        //         let value_loss_float = f32::try_from(&value_loss.detach()).unwrap();
        //         // if value_loss_float > 100. {
        //         //     let dbg = value_loss_float;
        //         // }
        //         val_loss.push(value_loss_float);
    
        //         let action_loss = -((&ratio * &advs).min_other(&(&clip_ratio * &advs)).mean(Kind::Float));
        //         let action_loss_float = f32::try_from(&action_loss.detach()).unwrap();
        //         act_loss.push(action_loss_float);
                
        //         // only for stats purposes at this time
        //         let loss = value_loss + &action_loss - &dist_entropy * entropy_coef;

        //         let full_act_loss = action_loss - dist_entropy * entropy_coef;
        //         losses.push(f32::try_from(&loss.detach()).unwrap());
        //         opt_act.backward_step_clip_norm(&full_act_loss, grad_clip);
        //         opt_critic.backward_step_clip_norm(value_loss, grad_clip);
        //     }
        // }
        // prog_bar.finish_and_clear();
        // if update_index > 0 && update_index % 25 == 0 {
        //     let tot = clip_fracs.iter().sum::<f32>();
        //     let clip_frac = tot / clip_fracs.len() as f32;

        //     let tot = kl_divs.iter().sum::<f32>();
        //     let kl_div = tot / kl_divs.len() as f32;
            
        //     let tot = entropys.iter().sum::<f32>();
        //     let entropy = tot / entropys.len() as f32;

        //     let tot = losses.iter().sum::<f32>();
        //     let loss = tot / losses.len() as f32;

        //     let tot = act_loss.iter().sum::<f32>();
        //     let act_l = tot / act_loss.len() as f32;

        //     let tot = val_loss.iter().sum::<f32>();
        //     let val_l = tot / val_loss.len() as f32;

            println!("update idx: {}, total eps: {:.0}, episode rewards: {}, total steps: {}",
             update_index, total_episodes, total_rewards / total_episodes, total_steps);
            total_rewards = 0.;
            total_episodes = 0.;
        // }
        if update_index > 0 && update_index % 1000 == 0 {
            // NOTE: the ValueStore doesn't seem to store the optimizer state (which seems to be bound to the optimizer itself)
            // if let Err(err) = vs.save(format!("trpo{update_index}.ot")) {
            //     println!("error while saving {err}")
            // }
        }
        // println!("\nnext set -------------------\n");
    }
    // total_prog_bar.finish_and_clear();
    // Ok(())
}