use bytebuffer::ByteBuffer;
use serde::{Deserialize, Serialize};
use tch::{Tensor, TchError, nn::VarStore};



// const ROLLOUT_KEY: &str = "rollouts";


pub fn tensor_to_bytes(tensor: &Tensor) -> Result<Vec<u8>, TchError> {
    let mut buffer = ByteBuffer::new();

    let op = tensor.save_to_stream(&mut buffer);

    match op {
        Ok(_) => Ok(buffer.into_vec()),
        Err(e) => Err(e)
    }
}

pub fn varstore_to_bytes(vs: &VarStore) -> Result<Vec<u8>, TchError> {
    let mut buffer = ByteBuffer::new();

    let op = vs.save_to_stream(&mut buffer);

    match op {
        Ok(_) => Ok(buffer.into_vec()),
        Err(e) => Err(e)
    }
}

#[derive(Serialize, Deserialize)]
pub struct ExperienceStoreProcs {
    pub s_states: Vec<Vec<Vec<f32>>>,
    pub s_rewards: Vec<Vec<f32>>,
    pub s_actions: Vec<Vec<f32>>,
    pub dones_f: Vec<Vec<f32>>,
    pub s_log_probs: Vec<Vec<f32>>,
}

#[derive(Serialize, Deserialize)]
pub struct ExperienceStore {
    pub s_states: Vec<Vec<f32>>,
    pub s_rewards: Vec<f32>,
    pub s_actions: Vec<f32>,
    pub dones_f: Vec<f32>,
    pub s_log_probs: Vec<f32>,
}

// pub struct ExperienceStoreTen {
//     pub s_states: Tensor,
//     pub s_rewards: Tensor,
//     pub s_actions: Tensor,
//     pub dones_f: Tensor,
//     pub s_log_probs: Tensor,
// }

// pub struct ExpAdvHolder {
//     pub s_states: Vec<Vec<f32>>,
//     pub s_rewards: Vec<f32>,
//     pub s_actions: Vec<f32>,
//     pub dones_f: Vec<f32>,
//     pub s_log_probs: Vec<f32>,
//     pub advs: Vec<f32>,
//     pub targ_vals: Vec<f32>,
// }