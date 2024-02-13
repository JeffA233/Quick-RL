pub mod models;
pub mod algorithms;

/* Proximal Policy Optimization (PPO) model.

   Proximal Policy Optimization Algorithms, Schulman et al. 2017
   https://arxiv.org/abs/1707.06347

   See https://spinningup.openai.com/en/latest/algorithms/ppo.html for a
   reference python implementation.
*/
pub mod vec_gym_env;
pub mod gym_lib;
pub mod gym_funcs;
pub mod tch_utils;
// use bytebuffer::ByteBuffer;
// use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
// use tch::nn::init::{NonLinearity, NormalOrUniform};
// use vec_gym_env::VecGymEnv;
// use tch::kind::{FLOAT_CPU, INT64_CPU};
// use tch::{nn::{self, init, LinearConfig, OptimizerConfig}, Device, Kind, Tensor};

// use crate::{
//     algorithms::common_utils::gather_experience::ppo_gather::get_experience, 
//     models::{model_base::{DiscreteActPPO, Model}, ppo::default_ppo::{Actor, Critic, LayerConfig}}, 
//     // tch_utils::dbg_funcs::{
//     //     print_tensor_2df32, 
//     //     print_tensor_noval, 
//     //     print_tensor_vecf32
//     // }
// };