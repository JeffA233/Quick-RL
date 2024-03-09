use bytebuffer::ByteBuffer;
use serde::{Deserialize, Serialize};
use tch::{nn::VarStore, TchError, Tensor};

// const ROLLOUT_KEY: &str = "rollouts";

pub fn tensor_to_bytes(tensor: &Tensor) -> Result<Vec<u8>, TchError> {
    let mut buffer = ByteBuffer::new();

    let op = tensor.save_to_stream(&mut buffer);

    match op {
        Ok(_) => Ok(buffer.into_vec()),
        Err(e) => Err(e),
    }
}

pub fn varstore_to_bytes(vs: &VarStore) -> Result<Vec<u8>, TchError> {
    let mut buffer = ByteBuffer::new();

    let op = vs.save_to_stream(&mut buffer);

    match op {
        Ok(_) => Ok(buffer.into_vec()),
        Err(e) => Err(e),
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
    pub terminal_obs: Vec<f32>,
    pub model_ver: i64,
}

pub trait RolloutDatabaseBackend {
    fn get_key_value_i64(&mut self, key: &str) -> Result<i64, Box<dyn std::error::Error>>;
    fn get_key_value_bool(&mut self, key: &str) -> Result<bool, Box<dyn std::error::Error>>;
    fn get_key_value_raw(&mut self, key: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>>;
    fn rpush(&mut self, key: &str, value: &[u8]) -> Result<(), Box<dyn std::error::Error>>;
    fn get_experience(&mut self, num_steps: usize, min_ver: i64) -> ExperienceStore;
    fn set_key_value(
        &mut self,
        key: &str,
        value: impl Serialize,
    ) -> Result<(), Box<dyn std::error::Error>>;
    fn set_key_value_raw(
        &mut self,
        key: &str,
        value: &[u8],
    ) -> Result<(), Box<dyn std::error::Error>>;
    fn set_key_value_i64(
        &mut self,
        key: &str,
        value: i64,
    ) -> Result<(), Box<dyn std::error::Error>>;
    fn set_key_value_bool(
        &mut self,
        key: &str,
        value: bool,
    ) -> Result<(), Box<dyn std::error::Error>>;
    fn del(&mut self, key: &str) -> Result<(), Box<dyn std::error::Error>>;
    fn incr(&mut self, key: &str, increment: i64) -> Result<i64, Box<dyn std::error::Error>>;
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
