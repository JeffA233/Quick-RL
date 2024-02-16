use std::time::Duration;

use redis::{Client, Commands, Connection};
use serde::Serialize;

use super::rollout_buffer_utils::ExperienceStore;


// pub struct RolloutBufferWorker {
//     // redis_url: String,
//     redis_con: Connection,
//     states: Vec<Vec<f32>>,
//     rewards: Vec<f32>,
//     // NOTE: actions is only this format with discrete, with multidiscrete or continuous it will not be
//     // we might want to change or update this in the future
//     actions: Vec<f32>,
//     dones: Vec<f32>,
//     log_probs: Vec<f32>,
// }

// impl RolloutBufferWorker {
//     pub fn new(redis_url: String) -> Self {
//         let redis_client = Client::open(redis_url).unwrap();
//         let redis_con = redis_client.get_connection_with_timeout(Duration::from_secs(30)).unwrap();
//         Self {
//             // redis_url,
//             redis_con,
//             states: Vec::new(),
//             rewards: Vec::new(),
//             actions: Vec::new(),
//             dones: Vec::new(),
//             log_probs: Vec::new(),
//         }
//     }
    
//     /// note here that the state is actually the previous state t+0 from the gym and not the current one which is t+1
//     pub fn push_experience(&mut self, state: Vec<f32>, reward: f32, action: f32, done: f32, log_prob: f32) {
//         self.states.push(state);
//         self.rewards.push(reward);
//         self.actions.push(action);
//         self.dones.push(done);
//         self.log_probs.push(log_prob);
//     }

//     pub fn submit_rollout(mut self) {  
//         let experience = ExperienceStore { s_states: self.states, s_rewards: self.rewards, s_actions: self.actions, dones_f: self.dones, s_log_probs: self.log_probs };
//         // serialize data
//         let mut s = flexbuffers::FlexbufferSerializer::new();
//         experience.serialize(&mut s).unwrap();
//         let view = s.view();
//         // rpush experience
//         self.redis_con.rpush::<&str, &[u8], ()>("exp_store", view).unwrap();

//         self.states = Vec::new();
//         self.rewards = Vec::new();
//         self.actions = Vec::new();
//         self.dones = Vec::new();
//         self.log_probs = Vec::new();
//     }
// }

/// Convenience worker to deal with the tensor workflow while testing tch
pub struct RolloutBufferWorkerTen {
    // redis_url: String,
    redis_con: Connection,
}

impl RolloutBufferWorkerTen {
    pub fn new(redis_url: String) -> Self {
        let redis_client = Client::open(redis_url).unwrap();
        let redis_con = redis_client.get_connection_with_timeout(Duration::from_secs(30)).unwrap();
        Self {
            // redis_url,
            redis_con,
        }
    }
    
    /// note here that the state is actually the previous state t+0 from the gym and not the current one which is t+1
    // pub fn push_experience(&mut self, state: Vec<f32>, reward: f32, action: f32, done: f32, log_prob: f32) {
    //     self.states.push(state);
    //     self.rewards.push(reward);
    //     self.actions.push(action);
    //     self.dones.push(done);
    //     self.log_probs.push(log_prob);
    // }

    pub fn submit_rollout(&mut self, experience: ExperienceStore) {  
        // serialize data
        let mut s = flexbuffers::FlexbufferSerializer::new();
        experience.serialize(&mut s).unwrap();
        let view = s.view();
        // let view_dbg = view.to_vec();
        // rpush experience
        self.redis_con.rpush::<&str, Vec<u8>, ()>("exp_store", view.to_vec()).unwrap();
    }
}