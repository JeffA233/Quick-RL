use std::{env, ffi::OsString, path::PathBuf};

/* Proximal Policy Optimization (PPO) model.

   Proximal Policy Optimization Algorithms, Schulman et al. 2017
   https://arxiv.org/abs/1707.06347

   See https://spinningup.openai.com/en/latest/algorithms/ppo.html for a
   reference python implementation.
*/
use bytebuffer::ByteBuffer;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde::Serialize;
// use tch::nn::init::{NonLinearity, NormalOrUniform};
// use quick_rl::vec_gym_env::VecGymEnv;
// use tch::kind::{FLOAT_CPU, INT64_CPU};
use tch::{
    nn::{self, init, LinearConfig, OptimizerConfig},
    Device, Kind, Tensor,
};

use quick_rl::{
    algorithms::{
        common_utils::{
            // rollout_buffer::rollout_buffer_host::RolloutBufferHost,
            rollout_buffer::rollout_buffer_host::{RedisRolloutBackend, RolloutHostBackend},
            GAECalc,
        },
        ppo::ppo_learn::PPOLearner,
    },
    config::Configuration,
    models::{
        model_base::Model,
        ppo::default_ppo::{Actor, Critic, LayerConfig},
    },
    // tch_utils::dbg_funcs::{
    //     print_tensor_2df32,
    //     print_tensor_noval,
    //     print_tensor_vecf32
    // }
    vec_gym_env::VecGymEnv,
};

pub fn main() {
    // NOTE:
    // rough benchmark is ~4.26 for rew by idx 150
    // --- env setup stuff ---
    // I hate this path stuff but I'm not sure what's cleaner
    let mut config_path = PathBuf::new();
    config_path.push("src/config.json");
    let config = match Configuration::load_configuration(config_path.as_path()) {
        Ok(config) => config,
        Err(error) => {
            panic!(
                "Error loading configuration from '{}': {}",
                config_path.display(),
                error
            );
        }
    };

    let device = if config.device.to_lowercase() == "cuda" {
        Device::cuda_if_available()
    } else {
        Device::Cpu
    };
    let reward_file_full_path = config.reward_file_full_path.clone();
    let updates = config.hyperparameters.updates;

    tch::manual_seed(0);
    tch::Cuda::manual_seed_all(0);

    // how old a model can be, logic is (current_ver - min_model_ver) < rollout_model_ver else discard step
    // TL;DR 1 means it could be data from the last model theoretically and so on
    let max_model_age = config.hyperparameters.max_model_age;

    // configure number of agents and gamemodes
    let mut team_size = Vec::new();
    team_size.extend(vec![1; config.gamemodes.num_1s]);
    team_size.extend(vec![2; config.gamemodes.num_2s]);
    team_size.extend(vec![3; config.gamemodes.num_3s]);

    let mut self_plays = Vec::new();
    self_plays.extend(vec![
        false;
        config.gamemodes.num_1s
            - config.gamemodes.num_1s_selfplay
    ]);
    self_plays.extend(vec![true; config.gamemodes.num_1s_selfplay]);
    self_plays.extend(vec![
        false;
        config.gamemodes.num_2s
            - config.gamemodes.num_2s_selfplay
    ]);
    self_plays.extend(vec![true; config.gamemodes.num_2s_selfplay]);
    self_plays.extend(vec![
        false;
        config.gamemodes.num_3s
            - config.gamemodes.num_3s_selfplay
    ]);
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
    let env = VecGymEnv::new(
        team_size,
        self_plays,
        config.tick_skip,
        reward_file_full_path,
    );
    println!("action space: {}", env.action_space());
    let obs_space = env.observation_space()[1];
    println!("observation space: {:?}", obs_space);

    // get Redis connection
    let db = config.redis.dbnum.clone();
    let password = if !config.redis.password_env_var.is_empty() {
        env::var_os(&config.redis.password_env_var).expect("Failed to get password")
    } else {
        OsString::from("")
    };
    let password_str = password
        .to_str()
        .expect("Failed to convert password to str");
    let redis_address = config.redis.ipaddress;
    let redis_str = format!(
        "redis://{}:{}@{}/{}",
        config.redis.username, password_str, redis_address, db
    );
    let redis_str = redis_str.as_str();
    // change to make it generic
    let mut rollout_backend = RedisRolloutBackend::new(redis_str.to_string());

    // Create an instance of the generic RolloutBufferHost with the Redis backend
    // let mut buffer_host = Gen::new(redis_backend);

    // setup actor and critic
    let vs_act = nn::VarStore::new(device);
    let vs_critic = nn::VarStore::new(device);
    let init_config = Some(LinearConfig {
        ws_init: init::Init::Orthogonal { gain: 2_f64.sqrt() },
        bs_init: Some(init::Init::Const(0.)),
        bias: true,
    });
    let layer_vec_act = if !config.network.custom_shape {
        vec![config.network.actor.layer_size; config.network.actor.num_layers]
    } else {
        config.network.custom_actor.layer_vec
    };
    let act_config = LayerConfig::new(layer_vec_act, obs_space, Some(env.action_space()));
    let mut act_model = Actor::new(
        &vs_act.root(),
        act_config.clone(),
        init_config,
        config.network.act_func,
    );

    let layer_vec_critic = if !config.network.custom_shape {
        vec![config.network.critic.layer_size; config.network.critic.num_layers]
    } else {
        config.network.custom_critic.layer_vec
    };
    let critic_config = LayerConfig::new(layer_vec_critic, obs_space, None);
    let mut critic_model = Critic::new(&vs_critic.root(), critic_config, init_config);

    let mut opt_act = nn::Adam::default()
        .build(&vs_act, config.hyperparameters.lr)
        .unwrap();
    let mut opt_critic = nn::Adam::default()
        .build(&vs_critic, config.hyperparameters.lr)
        .unwrap();

    // trying to force worker to wait for model data
    rollout_backend.del("model_data").unwrap();

    rollout_backend.set_key_value_i64("model_ver", 0).unwrap();
    // use this flag to pause episode gathering if on the same PC, just testing for now
    // we should make this toggleable probably
    rollout_backend
        .set_key_value_bool("gather_pause", false)
        .unwrap();
    let mut model_ver: i64 = 0;

    // send layer config to worker(s) for tch/libtorch usage
    let mut s = flexbuffers::FlexbufferSerializer::new();
    act_config.serialize(&mut s).unwrap();
    rollout_backend
        .set_key_value_raw("actor_structure", s.view())
        .unwrap();

    // misc stats stuff
    let mut total_episodes = 0f64;
    let mut total_steps = 0i64;

    let n_steps = config.hyperparameters.steps_per_rollout;
    let n_procs = config.n_env;
    let train_size = n_steps * n_procs;
    let buffersize = config.hyperparameters.buffersize;
    let optim_batchsize = buffersize as i64;
    let optim_epochs = config.hyperparameters.optim_epochs;

    let gae_calc = GAECalc::new(
        Some(config.hyperparameters.gamma),
        Some(config.hyperparameters.lambda),
    );
    // let train_size = buffersize as i64;
    let ppo_learner = PPOLearner::new(
        optim_epochs,
        optim_batchsize as usize,
        config.hyperparameters.clip_range,
        config.hyperparameters.entropy_coef,
        config.hyperparameters.grad_clip,
        device,
        train_size,
    );

    // start of learning loops
    for update_index in 0..config.hyperparameters.updates {
        let mut act_save_stream = ByteBuffer::new();
        // TODO: consider parsing this result
        vs_act.save_to_stream(&mut act_save_stream).unwrap();
        let act_buffer = act_save_stream.into_vec();

        rollout_backend.incr("model_ver", 1).unwrap();
        model_ver += 1;

        rollout_backend
            .set_key_value_raw("model_data", &act_buffer)
            .unwrap();
        // clear redis
        rollout_backend.del("exp_store").unwrap();

        rollout_backend
            .set_key_value_bool("gather_pause", false)
            .unwrap();

        total_steps += n_steps * n_procs;

        let mut exp_store = rollout_backend.get_experience(buffersize, model_ver - max_model_age);
        println!("gathered rollouts");
        rollout_backend
            .set_key_value_bool("gather_pause", true)
            .unwrap();

        exp_store.s_states.push(exp_store.terminal_obs);

        // move to Tensor
        let s_actions = Tensor::from_slice(&exp_store.s_actions);
        let states = Tensor::from_slice2(&exp_store.s_states);
        // println!("states size was: {}", ten_vec.len());
        // println!("states size 2 was: {}", ten_vec[0].len());

        let s_log_probs = Tensor::from_slice(&exp_store.s_log_probs);
        let s_rewards = Tensor::from_slice(&exp_store.s_rewards);
        let dones_f = Tensor::from_slice(&exp_store.dones_f);
        //
        // print_tensor_noval("states", &s_states);  // size = [1025, 16, 107]
        // print_tensor_noval("rewards", &s_rewards);
        // print_tensor_noval("actions", &s_actions);
        // print_tensor_noval("dones", &dones_f);
        // print_tensor_noval("log_probs", &s_log_probs);
        // print_tensor_noval("states after view", &states);

        // compute gae
        let vals = tch::no_grad(|| {
            critic_model.forward(&states.to_device_(device, Kind::Float, true, false))
        })
        .squeeze()
        .to_device_(Device::Cpu, Kind::Float, true, false);
        let adv = gae_calc.calc(&s_rewards, &dones_f, &vals);
        // print_tensor_noval("vals from critic", &vals);

        // shrink everything to fit batch size
        // we want to do this after GAE in order to make sure we use the full rollout data that we are given for a more accurate GAE calc
        let advantages = adv.narrow(0, 0, n_steps * n_procs).view([train_size, 1]);

        // we now must do view on everything we want to batch
        let target_vals = (&advantages
            + vals.narrow(0, 0, n_steps * n_procs).view([train_size, 1]))
        .to_device_(device, Kind::Float, true, false);

        let learn_states = states
            .narrow(0, 0, n_steps * n_procs)
            .view([train_size, config.n_stack, obs_space])
            .to_device_(device, Kind::Float, true, false);

        // norm advantages
        let advantages = ((&advantages - &advantages.mean(Kind::Float))
            / (&advantages.std(true) + 1e-8))
            .to_device_(device, Kind::Float, true, false);

        let actions = s_actions
            .narrow(0, 0, n_steps * n_procs)
            .view([train_size])
            .to_device_(device, Kind::Int64, true, false);
        let old_log_probs = s_log_probs
            .narrow(0, 0, n_steps * n_procs)
            .view([train_size])
            .to_device_(device, Kind::Float, true, false);

        let prog_bar = multi_prog_bar_total.add(prog_bar_func((optim_epochs) as u64));
        let (clip_fracs, kl_divs, entropys, losses, act_loss, val_loss) = ppo_learner.do_calc(
            &mut act_model,
            &mut opt_act,
            &mut critic_model,
            &mut opt_critic,
            &actions,
            &advantages,
            &target_vals,
            &old_log_probs,
            &learn_states,
            &prog_bar,
        );

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
        //             // for viewing dbg values
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
        //         // dbg
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

            // println!("update idx: {}, total eps: {:.0}, episode rewards: {}, total steps: {}, clip frac avg: {}, kl div avg: {}, ent: {}, loss: {}, act loss: {}, val loss: {}",
            println!("update idx: {}, total eps: {:.0}, total steps: {}, clip frac avg: {}, kl div avg: {}, ent: {}, loss: {}, act loss: {}, val loss: {}",
            //  update_index, total_episodes, total_rewards / total_episodes, total_steps, clip_frac, kl_div, entropy, loss, act_l, val_l);
            update_index, total_episodes, total_steps, clip_frac, kl_div, entropy, loss, act_l, val_l);
            // total_rewards = 0.;
            total_episodes = 0.;
        }
        if update_index > 0 && update_index % 1000 == 0 {
            // NOTE: the ValueStore doesn't seem to store the optimizer state (which seems to be bound to the optimizer itself)
            // if let Err(err) = vs.save(format!("trpo{update_index}.ot")) {
            //     println!("error while saving {err}")
            // }
        }
        // println!("\nnext set -------------------\n");
    }
    total_prog_bar.finish_and_clear();
}
