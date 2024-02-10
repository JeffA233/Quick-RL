pub mod models;
pub mod algorithms;

/* Proximal Policy Optimization (PPO) model.

   Proximal Policy Optimization Algorithms, Schulman et al. 2017
   https://arxiv.org/abs/1707.06347

   See https://spinningup.openai.com/en/latest/algorithms/ppo.html for a
   reference python implementation.
*/
mod vec_gym_env;
pub mod gym_lib;
pub mod gym_funcs;
pub mod tch_utils;
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
// use tch::nn::init::{NonLinearity, NormalOrUniform};
use vec_gym_env::VecGymEnv;
// use tch::kind::{FLOAT_CPU, INT64_CPU};
use tch::{nn::{self, init, LinearConfig, OptimizerConfig}, Device, Kind, Tensor};

use crate::{algorithms::common_utils::gather_experience::ppo_gather::get_experience, models::{model_base::{DiscreteActPPO, Model}, ppo::default_ppo::{Actor, Critic}}, tch_utils::dbg_funcs::{print_tensor_2df32, print_tensor_noval, print_tensor_vecf32}};

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
const BUFFERSIZE: i64 = NSTEPS*NPROCS;
// const OPTIM_BATCHSIZE: i64 = 6;
const OPTIM_BATCHSIZE: i64 = BUFFERSIZE/1;
const OPTIM_EPOCHS: i64 = 20;

// type Model = Box<dyn Fn(&Tensor) -> (Tensor, Tensor)>;
// type ModelActCritic = Box<dyn Fn(&Tensor) -> Tensor>;

// fn model(p: &nn::Path, nact: i64, n_in: i64) -> Model {
//     // layer_config
//     let num_layers = 4;
//     let net_dim = 256;
//     // define layer functions
    // let layer_func = |in_dim: i64, out_dim: i64, layer_name: String| nn::linear(p / layer_name, in_dim, out_dim, Default::default());
//     let activation_func = |xs: &Tensor| xs.relu();
//     // let init_func = || nn::LinearConfig{ws_init: Kaiming{dist: Normal, fan: nn::init::FanInOut::FanIn, non_linearity: NonLinearity::ReLU},bs_init:Some(Kaiming{dist: Normal,fan: FanIn,non_linearity: ReLU}),bias:true};
    // let init_config_func = || Init::Kaiming{dist: nn::init::NormalOrUniform::Normal, fan: init::FanInOut::FanIn, non_linearity: init::NonLinearity::ReLU};
//     // start building network
//     let seq = nn::seq();
//     seq.add(layer_func(n_in, net_dim, String::from("l0")));
//     seq.add_fn(|xs| activation_func(xs));
//     for i in 1..num_layers {
//         let layer_str = String::from("l") + &i.to_string();
//         seq.add(layer_func(net_dim, net_dim, layer_str));
//         seq.add_fn(|xs| activation_func(xs));
//     }
//     let critic = nn::linear(p / "cl", 256, 1, Default::default());
//     let actor = nn::linear(p / "al", 256, nact, Default::default());
//     let device = p.device();
//     Box::new(move |xs: &Tensor| {
//         let xs = xs.to_device(device).apply(&seq);
//         (xs.apply(&critic), xs.apply(&actor))
//     })
// }

// fn actor_model(p: &nn::Path, nact: i64, n_in: i64) -> ModelActCritic {
//     // layer_config
//     let num_layers = 4;
//     let net_dim = 256;
//     // define layer functions
//     let layer_func = |in_dim: i64, out_dim: i64, layer_str: String| nn::linear(p / layer_str, in_dim, out_dim, Default::default());
//     let activation_func = |xs: &Tensor| xs.relu();
//     // start building network
//     let mut seq = nn::seq();
//     seq = seq.add(layer_func(n_in, net_dim, String::from("l0")));
//     seq = seq.add_fn(activation_func);
//     for i in 1..num_layers {
//         let layer_str = String::from("l") + &i.to_string();
//         seq = seq.add(layer_func(net_dim, net_dim, layer_str));
//         seq = seq.add_fn(activation_func);
//     }
//     let actor = nn::linear(p / "al", 256, nact, Default::default());
//     let device = p.device();
//     Box::new(move |xs: &Tensor| {
//         assert!(xs.device() == device, "Tensor in actor was on wrong device: {:#?}", xs.device());
//         // let xs = xs.to_device(device).apply(&seq);
//         let xs = xs.apply(&seq);
//         xs.apply(&actor)
//     })
// }

// fn critic_model(p: &nn::Path, n_in: i64) -> ModelActCritic {
//     // layer_config
//     let num_layers = 4;
//     let net_dim = 256;
//     // define layer functions
//     let layer_func = |in_dim: i64, out_dim: i64, layer_str: String| nn::linear(p / layer_str, in_dim, out_dim, Default::default());
//     let activation_func = |xs: &Tensor| xs.relu();
//     // start building network
//     let mut seq = nn::seq();
//     seq = seq.add(layer_func(n_in, net_dim, String::from("l0")));
//     seq = seq.add_fn(activation_func);
//     for i in 1..num_layers {
//         let layer_str = String::from("l") + &i.to_string();
//         seq = seq.add(layer_func(net_dim, net_dim, layer_str));
//         seq = seq.add_fn(activation_func);
//     }
//     let critic = nn::linear(p / "cl", 256, 1, Default::default());
//     let device = p.device();
//     Box::new(move |xs: &Tensor| {
//         assert!(xs.device() == device, "Tensor in critic was on wrong device: {:#?}", xs.device());
//         // let xs = xs.to_device(device).apply(&seq);
//         let xs = xs.apply(&seq);
//         xs.apply(&critic)
//     })
// }

// #[derive(Debug)]
// struct FrameStack {
//     data: Tensor,
//     nprocs: i64,
//     nstack: i64,
// }

// impl FrameStack {
//     fn new(nprocs: i64, nstack: i64) -> FrameStack {
//         FrameStack { data: Tensor::zeros([nprocs, nstack, 347], FLOAT_CPU), nprocs, nstack }
//     }

//     fn update<'a>(&'a mut self, img: &Tensor, masks: Option<&Tensor>) -> &'a Tensor {
//         if let Some(masks) = masks {
//             self.data *= masks.view([self.nprocs, 1, 1])
//         };
//         let slice = |i| self.data.narrow(1, i, 1);
//         for i in 1..self.nstack {
//             slice(i - 1).copy_(&slice(i))
//         }
//         slice(self.nstack - 1).copy_(img);
//         &self.data
//     }
// }

pub fn main() {
    // --- env setup stuff ---
    let tick_skip = 8;
    let entropy_coef = 0.01;
    let clip_range = 0.2;
    let grad_clip = 0.5;
    let lr = 5e-4;
    let gamma = 0.99;
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
    // --- end of setup ---
    let mut env = VecGymEnv::new(match_nums, gravity_nums, boost_nums, self_plays, tick_skip, reward_file_name);
    println!("action space: {}", env.action_space());
    let obs_space = env.observation_space()[1];
    println!("observation space: {:?}", obs_space);

    let vs = nn::VarStore::new(device);
    let mut act_model = Actor::new(&vs.root(), env.action_space(), obs_space, vec![256; 3], Some(LinearConfig { ws_init: init::Init::Orthogonal { gain: 2_f64.sqrt() }, bs_init: Some(init::Init::Const(0.)), bias: true }));
    let mut critic_model = Critic::new(&vs.root(), obs_space, vec![256; 3], Some(LinearConfig { ws_init: init::Init::Orthogonal { gain: 2_f64.sqrt() }, bs_init: Some(init::Init::Const(0.)), bias: true }));
    let mut opt = nn::Adam::default().build(&vs, lr).unwrap();

    let mut sum_rewards = Tensor::zeros([NPROCS], (Kind::Float, Device::Cpu));
    let mut total_rewards = 0f64;
    let mut total_episodes = 0f64;
    let mut total_steps = 0i64;

    let train_size = NSTEPS * NPROCS;
    for update_index in 0..UPDATES {
        let (s_states, s_rewards, s_actions, dones_f, s_log_probs) = 
            get_experience(
                NSTEPS, 
                NPROCS, 
                obs_space, 
                device, 
                &multi_prog_bar_total, 
                &total_prog_bar, 
                prog_bar_func, 
                &mut act_model, 
                &mut env, 
                &mut sum_rewards, 
                &mut total_rewards, 
                &mut total_episodes
            );
        // -- 
        total_steps += NSTEPS * NPROCS;
        // print_tensor_noval("states", &s_states);  // size = [1025, 16, 107]
        // print_tensor_noval("rewards", &s_rewards);
        // print_tensor_noval("actions", &s_actions);
        // print_tensor_noval("dones", &dones_f);
        // print_tensor_noval("log_probs", &s_log_probs);
        let states = s_states.view([NSTEPS + 1, NPROCS, NSTACK, obs_space]);
        // print_tensor_noval("states after view", &states);

        // compute gae
        let adv = Tensor::zeros([NSTEPS, NPROCS], (Kind::Float, Device::Cpu));
        let vals = tch::no_grad(|| critic_model.forward(&states.to_device_(device, Kind::Float, true, false))).squeeze().to_device_(Device::Cpu, Kind::Float, true, false);
        // print_tensor_noval("vals from critic", &vals);
        let mut last_gae_lam = Tensor::zeros([NPROCS], (Kind::Float, Device::Cpu));
        for idx in (0..NSTEPS).rev() {
            let done = if idx == NSTEPS - 1 {
                1. - dones_f.get(idx)
            } else {
                1. - dones_f.get(idx + 1)
            };
            let next_val = vals.get(idx + 1);
            // print_tensor_f32("val", &val_idx);
            let rew = s_rewards.get(idx);
            // print_tensor_f32("rew", &rew);
            // print_tensor_noval("rew", &rew);
            // print_tensor_noval("next_val", &next_val);
            // print_tensor_noval("done", &done);
            let pred_ret = rew + gamma * &next_val * &done;
            // print_tensor_f32("pred_ret", &pred_ret);
            let delta = &pred_ret - &vals.get(idx);
            // print_tensor_f32("delta", &delta);
            last_gae_lam = delta + gamma * 0.95 * done * last_gae_lam;
            // print_tensor_f32("last_gae_lam", &last_gae_lam);
            adv.get(idx).copy_(&last_gae_lam);
            // print_tensor_f32("targ_val", &targ_val);
        }
        let advantages = adv.view([train_size, 1]);
        let target_vals = (&advantages + vals.narrow(0, 0, NSTEPS).view([train_size, 1])).to_device_(device, Kind::Float, true, false);

        let learn_states = states.narrow(0, 0, NSTEPS).view([train_size, NSTACK, obs_space]).to_device_(device, Kind::Float, true, false);

        // norm advantages
        let advantages = ((&advantages - &advantages.mean(Kind::Float)) / (&advantages.std(true) + 1e-8)).to_device_(device, Kind::Float, true, false);
        
        let actions = s_actions.view([train_size]).to_device_(device, Kind::Int64, true, false);
        let old_log_probs = s_log_probs.view([train_size]).to_device_(device, Kind::Float, true, false);
        let prog_bar = multi_prog_bar_total.add(prog_bar_func((OPTIM_EPOCHS) as u64));
        prog_bar.set_message("doing epochs");
        let mut clip_fracs = Vec::new();
        let mut kl_divs = Vec::new();
        let mut entropys = Vec::new();
        let mut losses = Vec::new();
        let mut act_loss = Vec::new();
        let mut val_loss = Vec::new();
        let optim_indexes = Tensor::randint(train_size, [OPTIM_EPOCHS, BUFFERSIZE], (Kind::Int64, device));
        for epoch in 0..OPTIM_EPOCHS {
            prog_bar.inc(1);
            let batch_indexes = optim_indexes.get(epoch);
            for batch_start_index in (0..BUFFERSIZE).step_by(OPTIM_BATCHSIZE as usize) {
                let buffer_indexes = batch_indexes.slice(0, batch_start_index, batch_start_index + OPTIM_BATCHSIZE, 1);
                let states = learn_states.index_select(0, &buffer_indexes);
                let actions = actions.index_select(0, &buffer_indexes);
                // print_tensor_vecf32("batch actions", &actions);
                let advs = advantages.index_select(0, &buffer_indexes).squeeze();
                // print_tensor_vecf32("batch advantages", &advs);
                let targ_vals = target_vals.index_select(0, &buffer_indexes).squeeze();
                // print_tensor_vecf32("batch targ vals", &targ_vals);
                let old_log_probs_batch = old_log_probs.index_select(0, &buffer_indexes).squeeze();
                // print_tensor_vecf32("batch old log probs", &old_log_probs_batch);
                // let acts = act_model(&states);
                let (action_log_probs, dist_entropy) = act_model.get_prob_entr(&states, &actions);
                let vals = critic_model.forward(&states).squeeze();

                // // print_tensor_vecf32("batch vals", &vals);
                // let probs = acts.softmax(-1, Kind::Float).view((-1, env.action_space()));
                // let log_probs = probs.clamp(1e-11, 1.).log();
                // let action_log_probs = {
                //     log_probs.gather(-1, &actions.unsqueeze(-1), false).squeeze()
                // };
                // // print_tensor_vecf32("action log probs", &action_log_probs);
                // let dist_entropy =
                //     -(&log_probs * &probs).sum_dim_intlist(-1, false, Kind::Float).mean(Kind::Float);
                let dist_entropy_float = tch::no_grad(|| {f32::try_from(&dist_entropy.detach()).unwrap()});
                entropys.push(dist_entropy_float);
    
                // PPO ratio
                let ratio = (&action_log_probs - &old_log_probs_batch).exp().squeeze();
                // print_tensor_vecf32("ratio", &ratio);
                let clip_ratio = ratio.clamp(1.0 - clip_range, 1.0 + clip_range);
                // print_tensor_vecf32("clip ratio", &clip_ratio);
                clip_fracs.push(tch::no_grad(|| {
                    let est = ((&ratio - 1.).abs().greater(clip_range).to_kind(Kind::Float)).mean(Kind::Float);
                    f32::try_from(&est.detach().to(Device::Cpu)).unwrap()
                }));
                kl_divs.push(tch::no_grad(|| {
                    let log_ratio = &action_log_probs - &old_log_probs_batch;
                    // let act_log_prob_mean = f64::try_from(&action_log_probs.mean(Kind::Float)).unwrap();
                    // let old_log_probs_batch_mean = f64::try_from(&old_log_probs_batch.mean(Kind::Float)).unwrap();
                    // let log_ratio_mean = f64::try_from(&log_ratio.mean(Kind::Float).detach()).unwrap();
                    let kl = (log_ratio.exp() - 1.) - log_ratio;
                    // for dbg
                    // act_log_prob_mean;
                    // old_log_probs_batch_mean;
                    // log_ratio_mean;
                    f32::try_from(kl.mean(Kind::Float).detach().to(Device::Cpu)).unwrap()
                }));
    
                let value_loss = &vals.mse_loss(&targ_vals.squeeze(), tch::Reduction::Mean).squeeze();
                let value_loss_float = f32::try_from(&value_loss.detach()).unwrap();
                // if value_loss_float > 100. {
                //     let dbg = value_loss_float;
                // }
                val_loss.push(value_loss_float);
    
                let action_loss = -((&ratio * &advs).min_other(&(&clip_ratio * &advs)).mean(Kind::Float));
                let action_loss_float = f32::try_from(&action_loss.detach()).unwrap();
                act_loss.push(action_loss_float);
    
                let loss = value_loss + action_loss - dist_entropy * entropy_coef;
                losses.push(f32::try_from(&loss.detach()).unwrap());
                opt.backward_step_clip_norm(&loss, grad_clip);
            }
        }
        prog_bar.finish_and_clear();
        if update_index > 0 && update_index % 25 == 0 {
            let tot = clip_fracs.iter().sum::<f32>();
            let clip_frac = tot / clip_fracs.len() as f32;

            let tot = kl_divs.iter().sum::<f32>();
            let kl_div = tot / kl_divs.len() as f32;
            
            let tot = entropys.iter().sum::<f32>();
            let entropy = tot / entropys.len() as f32;

            let tot = losses.iter().sum::<f32>();
            let loss = tot / losses.len() as f32;

            let tot = act_loss.iter().sum::<f32>();
            let act_l = tot / act_loss.len() as f32;

            let tot = val_loss.iter().sum::<f32>();
            let val_l = tot / val_loss.len() as f32;

            println!("update idx: {}, total eps: {:.0}, episode rewards: {}, total steps: {}, clip frac avg: {}, kl div avg: {}, ent: {}, loss: {}, act loss: {}, val loss: {}",
             update_index, total_episodes, total_rewards / total_episodes, total_steps, clip_frac, kl_div, entropy, loss, act_l, val_l);
            total_rewards = 0.;
            total_episodes = 0.;
        }
        if update_index > 0 && update_index % 1000 == 0 {
            // if let Err(err) = vs.save(format!("trpo{update_index}.ot")) {
            //     println!("error while saving {err}")
            // }
        }
        // println!("\nnext set -------------------\n");
    }
    total_prog_bar.finish_and_clear();
    // Ok(())
}