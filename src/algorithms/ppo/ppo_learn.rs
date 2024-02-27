use indicatif::ProgressBar;
use tch::{nn::Optimizer, Device, Kind, Tensor};

use crate::models::model_base::{DiscreteActPPO, CriticPPO};

pub struct PPOLearner {
    // policy: Box<dyn DiscreteActPPO>, 
    // critic: Box<dyn CriticPPO>, 
    n_epochs: i64, 
    optim_batch_size: usize, 
    // minibatch_size: usize, 
    clip_range: f64, 
    ent_coef: f64,
    // lr: f64,
    grad_clip: f64,
    device: Device,
    total_buffer_size: i64,
}

impl PPOLearner {
    pub fn new(
        n_epochs: i64, 
        batch_size: usize, 
        // minibatch_size: usize, 
        clip_range: f64, 
        ent_coef: f64, 
        // lr: f64, 
        grad_clip: f64, 
        device: Device, 
        final_buffer_size: i64,
    ) -> Self {
        Self {
            // policy,
            // critic,
            n_epochs,
            optim_batch_size: batch_size,
            // minibatch_size,
            clip_range,
            ent_coef,
            // lr,
            grad_clip,
            device,
            total_buffer_size: final_buffer_size,
        }
    }

    pub fn do_calc(
        &self, 
        policy: &mut dyn DiscreteActPPO, 
        policy_opt: &mut Optimizer, 
        critic: &mut dyn CriticPPO, 
        critic_opt: &mut Optimizer, 
        actions: &Tensor, 
        advantages: &Tensor, 
        target_vals: &Tensor, 
        old_log_probs: &Tensor, 
        learn_states: &Tensor,
        prog_bar: &ProgressBar,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        prog_bar.set_message("doing epochs");
        // learner metrics
        let mut clip_fracs = Vec::new();
        let mut kl_divs = Vec::new();
        let mut entropys = Vec::new();
        let mut losses = Vec::new();
        let mut act_loss = Vec::new();
        let mut val_loss = Vec::new();

        // generates randomized batch indices for training
        let optim_indexes = Tensor::randint(self.total_buffer_size, [self.n_epochs, self.total_buffer_size], (Kind::Int64, self.device));
        // learner epoch loop
        for epoch in 0..self.n_epochs {
            prog_bar.inc(1);
            let batch_indexes = optim_indexes.get(epoch);
            for batch_start_index in (0..self.total_buffer_size).step_by(self.optim_batch_size) {
                let buffer_indexes = batch_indexes.slice(0, batch_start_index, batch_start_index + self.optim_batch_size as i64, 1);
                let states = learn_states.index_select(0, &buffer_indexes);
                let actions = actions.index_select(0, &buffer_indexes);
                // print_tensor_vecf32("batch actions", &actions);
                let advs = advantages.index_select(0, &buffer_indexes).squeeze();
                // print_tensor_vecf32("batch advantages", &advs);
                let targ_vals = target_vals.index_select(0, &buffer_indexes).squeeze();
                // print_tensor_vecf32("batch targ vals", &targ_vals);
                let old_log_probs_batch = old_log_probs.index_select(0, &buffer_indexes).squeeze();
                // print_tensor_vecf32("batch old log probs", &old_log_probs_batch);
                let (action_log_probs, dist_entropy) = policy.get_prob_entr(&states, &actions);
                let vals = critic.forward(&states).squeeze();
                // // print_tensor_vecf32("batch vals", &vals);
                let dist_entropy_float = tch::no_grad(|| {f32::try_from(&dist_entropy.detach()).unwrap()});
                entropys.push(dist_entropy_float);
    
                // PPO ratio
                let ratio = (&action_log_probs - &old_log_probs_batch).exp().squeeze();
                // print_tensor_vecf32("ratio", &ratio);
                let clip_ratio = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range);
                // print_tensor_vecf32("clip ratio", &clip_ratio);
                clip_fracs.push(tch::no_grad(|| {
                    let est = ((&ratio - 1.).abs().greater(self.clip_range).to_kind(Kind::Float)).mean(Kind::Float);
                    f32::try_from(&est.detach().to(Device::Cpu)).unwrap()
                }));
                kl_divs.push(tch::no_grad(|| {
                    let log_ratio = &action_log_probs - &old_log_probs_batch;
                    // for viewing dbg values
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
                // dbg
                // if value_loss_float > 100. {
                //     let dbg = value_loss_float;
                // }
                val_loss.push(value_loss_float);
    
                let action_loss = -((&ratio * &advs).min_other(&(&clip_ratio * &advs)).mean(Kind::Float));
                let action_loss_float = f32::try_from(&action_loss.detach()).unwrap();
                act_loss.push(action_loss_float);
                
                // only for stats purposes at this time
                let loss = value_loss + &action_loss - &dist_entropy * self.ent_coef;

                let full_act_loss = action_loss - dist_entropy * self.ent_coef;
                losses.push(f32::try_from(&loss.detach()).unwrap());
                policy_opt.backward_step_clip_norm(&full_act_loss, self.grad_clip);
                critic_opt.backward_step_clip_norm(value_loss, self.grad_clip);
            }
        }

        (clip_fracs, kl_divs, entropys, losses, act_loss, val_loss)
    }
}