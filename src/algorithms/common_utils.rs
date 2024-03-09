use tch::{Device, Kind, Tensor};

pub mod gather_experience;
pub mod rollout_buffer;

pub struct GAECalc {
    gamma: f64,
    lambda: f64,
}

impl GAECalc {
    pub fn new(gamma: Option<f64>, lambda: Option<f64>) -> Self {
        Self {
            gamma: gamma.unwrap_or(0.99),
            lambda: lambda.unwrap_or(0.95),
        }
    }
    pub fn calc(&self, rewards: &Tensor, dones: &Tensor, vals: &Tensor) -> Tensor {
        assert!(
            rewards.kind() == Kind::Float,
            "rewards in gae calc was not of type float"
        );
        assert!(
            dones.kind() == Kind::Float,
            "dones in gae calc was not of type float"
        );
        assert!(
            vals.kind() == Kind::Float,
            "rewards in gae calc was not of type float"
        );

        let buf_size = rewards.size()[0];
        let adv = Tensor::zeros([buf_size], (Kind::Float, Device::Cpu));
        // let vals = tch::no_grad(|| critic_model.forward(&states.to_device_(device, Kind::Float, true, false))).squeeze().to_device_(Device::Cpu, Kind::Float, true, false);
        // print_tensor_noval("vals from critic", &vals);
        let mut last_gae_lam = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        for idx in (0..buf_size).rev() {
            let done = if idx == buf_size - 1 {
                1. - dones.get(idx)
            } else {
                1. - dones.get(idx + 1)
            };
            let next_val = vals.get(idx + 1);
            // print_tensor_f32("val", &val_idx);
            let rew = rewards.get(idx);
            // print_tensor_f32("rew", &rew);
            // print_tensor_noval("rew", &rew);
            // print_tensor_noval("next_val", &next_val);
            // print_tensor_noval("done", &done);
            let pred_ret = rew + self.gamma * &next_val * &done;
            // print_tensor_f32("pred_ret", &pred_ret);
            let delta = &pred_ret - &vals.get(idx);
            // print_tensor_f32("delta", &delta);
            last_gae_lam = delta + self.gamma * self.lambda * done * last_gae_lam;
            // print_tensor_f32("last_gae_lam", &last_gae_lam);
            adv.get(idx).copy_(&last_gae_lam.squeeze());
            // print_tensor_f32("targ_val", &targ_val);
        }

        adv
    }

    pub fn update_config(&mut self, gamma: Option<f64>, lambda: Option<f64>) {
        if gamma.is_some() {
            self.gamma = gamma.unwrap();
        }
        if lambda.is_some() {
            self.lambda = lambda.unwrap();
        }
    }
}

// pub fn gae_calc(rewards: &Tensor, dones: &Tensor, vals: &Tensor, gamma: f64) -> Tensor {
//     assert!(rewards.kind() == Kind::Float, "rewards in gae calc was not of type float");
//     assert!(dones.kind() == Kind::Float, "dones in gae calc was not of type float");
//     assert!(vals.kind() == Kind::Float, "rewards in gae calc was not of type float");

//     let buf_size = rewards.size()[0];
//     let adv = Tensor::zeros([buf_size], (Kind::Float, Device::Cpu));
//     // let vals = tch::no_grad(|| critic_model.forward(&states.to_device_(device, Kind::Float, true, false))).squeeze().to_device_(Device::Cpu, Kind::Float, true, false);
//     // print_tensor_noval("vals from critic", &vals);
//     let mut last_gae_lam = Tensor::zeros([1], (Kind::Float, Device::Cpu));
//     for idx in (0..buf_size).rev() {
//         let done = if idx == buf_size - 1 {
//             1. - dones.get(idx)
//         } else {
//             1. - dones.get(idx + 1)
//         };
//         let next_val = vals.get(idx + 1);
//         // print_tensor_f32("val", &val_idx);
//         let rew = rewards.get(idx);
//         // print_tensor_f32("rew", &rew);
//         // print_tensor_noval("rew", &rew);
//         // print_tensor_noval("next_val", &next_val);
//         // print_tensor_noval("done", &done);
//         let pred_ret = rew + gamma * &next_val * &done;
//         // print_tensor_f32("pred_ret", &pred_ret);
//         let delta = &pred_ret - &vals.get(idx);
//         // print_tensor_f32("delta", &delta);
//         last_gae_lam = delta + gamma * 0.95 * done * last_gae_lam;
//         // print_tensor_f32("last_gae_lam", &last_gae_lam);
//         adv.get(idx).copy_(&last_gae_lam.squeeze());
//         // print_tensor_f32("targ_val", &targ_val);
//     }

//     adv
// }
