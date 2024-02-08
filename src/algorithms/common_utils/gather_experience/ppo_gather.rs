use indicatif::{MultiProgress, ProgressBar};
use tch::{Device, Kind, Tensor};

use crate::{models::{model_base::DiscreteActPPO, ppo::default_ppo::Actor}, vec_gym_env::VecGymEnv};


pub fn get_experience(
    nsteps: i64, 
    nprocs: i64, 
    obs_space: i64, 
    device: Device, 
    multi_prog_bar_total: &MultiProgress, 
    total_prog_bar: &ProgressBar, 
    prog_bar_func: impl Fn(u64) -> ProgressBar,
    act_model: &mut Actor,
    env: &mut VecGymEnv,
    sum_rewards: &mut Tensor,
    total_rewards: &mut f64,
    total_episodes: &mut f64,
) -> (
    Tensor, Tensor, Tensor, Tensor, Tensor
) {
    let s_states = Tensor::zeros([nsteps + 1, nprocs, obs_space], (Kind::Float, device));
    s_states.get(0).copy_(&s_states.get(-1));
    let s_rewards = Tensor::zeros([nsteps, nprocs], (Kind::Float, Device::Cpu));
    let s_actions = Tensor::zeros([nsteps, nprocs], (Kind::Int64, Device::Cpu));
    let dones_f = Tensor::zeros([nsteps, nprocs], (Kind::Float, Device::Cpu));
    let s_log_probs = Tensor::zeros([nsteps, nprocs], (Kind::Float, Device::Cpu));
    // progress bar
    let prog_bar = multi_prog_bar_total.add(prog_bar_func((nsteps * nprocs) as u64));
    prog_bar.set_message("getting rollouts");
    // --
    for s in 0..nsteps {
        total_prog_bar.inc(nprocs as u64);
        prog_bar.inc(nprocs as u64);

        let (actions, log_prob) = tch::no_grad(|| act_model.get_act_prob(&s_states.get(s).to_device_(device, Kind::Float, true, false), false));
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
        *total_rewards +=
            f64::try_from((&*sum_rewards * &step.is_done).sum(Kind::Float)).unwrap();
        *total_episodes += f64::try_from(step.is_done.sum(Kind::Float)).unwrap();

        let masks = &step.is_done.to_kind(Kind::Float);
        *sum_rewards *= &step.is_done.bitwise_not();
        s_actions.get(s).copy_(&actions_sqz);
        s_states.get(s + 1).copy_(&step.obs);
        s_rewards.get(s).copy_(&step.reward);
        dones_f.get(s).copy_(masks);
        s_log_probs.get(s).copy_(&log_prob_flat);
    }

    prog_bar.finish_and_clear();

    (s_states, s_rewards, s_actions, dones_f, s_log_probs)
}