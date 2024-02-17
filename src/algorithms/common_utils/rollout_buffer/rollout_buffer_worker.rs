use std::time::Duration;

use redis::{Client, Commands, Connection};
use serde::Serialize;

use super::rollout_buffer_utils::ExperienceStore;


pub struct RolloutBufferWorker {
    // redis_url: String,
    redis_con: Connection,
    states: Vec<Vec<f32>>,
    rewards: Vec<f32>,
    // NOTE: actions is only this format with discrete, with multidiscrete or continuous it will not be
    // we might want to change or update this in the future
    actions: Vec<f32>,
    dones: Vec<f32>,
    log_probs: Vec<f32>,
}

impl RolloutBufferWorker {
    pub fn new(redis_url: String, obs_space: i64, nsteps: i64) -> Self {
        let redis_client = Client::open(redis_url).unwrap();
        let redis_con = redis_client.get_connection_with_timeout(Duration::from_secs(30)).unwrap();
        let mut states = Vec::with_capacity(nsteps as usize);
        states.push(vec![0.; obs_space as usize]);
        Self {
            // redis_url,
            redis_con,
            states,
            rewards: Vec::with_capacity(nsteps as usize),
            actions: Vec::with_capacity(nsteps as usize),
            dones: Vec::with_capacity(nsteps as usize),
            log_probs: Vec::with_capacity(nsteps as usize),
        }
    }
    
    /// note here that the state is actually the previous state t+0 from the gym and not the current one which is t+1
    pub fn push_experience(&mut self, state: Vec<f32>, reward: f32, action: f32, done: f32, log_prob: f32) -> bool {
        self.states.push(state);
        // self.states.iter_mut().zip(state).map(|(vec, val)| vec.push(val)).for_each(drop);
        self.rewards.push(reward);
        // self.rewards.iter_mut().zip(reward).map(|(vec, val)| vec.push(val)).for_each(drop);
        self.actions.push(action);
        // self.actions.iter_mut().zip(action).map(|(vec, val)| vec.push(val)).for_each(drop);
        self.dones.push(done);
        // self.dones.iter_mut().zip(done).map(|(vec, val)| vec.push(val)).for_each(drop);
        self.log_probs.push(log_prob);
        // self.log_probs.iter_mut().zip(log_prob).map(|(vec, val)| vec.push(val)).for_each(drop);
        // for (i, done) in done.iter().enumerate() {
        if done == 1. {
            let last_state = self.states.pop().unwrap();
            // new_state_vec.push(last_state.clone());
            assert!(self.states.len() == self.rewards.len(), "states length was not correct compared to rewards in exp gather func: {}, {}", self.states.len(), self.rewards.len());
            // TODO: shouldn't need to clone here, feels dumb
            self.submit_rollout(ExperienceStore { s_states: self.states.clone(), s_rewards: self.rewards.clone(), s_actions: self.actions.clone(), dones_f: self.dones.clone(), s_log_probs: self.log_probs.clone(), terminal_obs: last_state.clone() });
            // start new episodes
            self.rewards.clear();
            self.actions.clear();
            self.dones.clear();
            self.log_probs.clear();
            self.states.clear();

            // let mut new_state_vec = Vec::new();
            // let last_state = s_states.last().unwrap().clone();
            // new_state_vec.push(last_state);
            self.states.push(last_state);

            // if s > nsteps {
            //     env_done_stores[i] = true;
            // }
            return true
        }
        false
        // }
    }

    fn submit_rollout(&mut self, experience: ExperienceStore) {
        // let last_state = self.states[i].pop().unwrap();
        // let experience = ExperienceStore { s_states: self.states, s_rewards: self.rewards, s_actions: self.actions, dones_f: self.dones, s_log_probs: self.log_probs };
        // serialize data
        let mut s = flexbuffers::FlexbufferSerializer::new();
        experience.serialize(&mut s).unwrap();
        let view = s.view();
        // rpush experience
        self.redis_con.rpush::<&str, &[u8], ()>("exp_store", view).unwrap();

        // self.states = Vec::new();
        // self.rewards = Vec::new();
        // self.actions = Vec::new();
        // self.dones = Vec::new();
        // self.log_probs = Vec::new();
    }
}

// /// Convenience worker to deal with the tensor workflow while testing tch
// pub struct RolloutBufferWorkerTen {
//     // redis_url: String,
//     redis_con: Connection,
// }

// impl RolloutBufferWorkerTen {
//     pub fn new(redis_url: String) -> Self {
//         let redis_client = Client::open(redis_url).unwrap();
//         let redis_con = redis_client.get_connection_with_timeout(Duration::from_secs(30)).unwrap();
//         Self {
//             // redis_url,
//             redis_con,
//         }
//     }
    
//     /// note here that the state is actually the previous state t+0 from the gym and not the current one which is t+1
//     // pub fn push_experience(&mut self, state: Vec<f32>, reward: f32, action: f32, done: f32, log_prob: f32) {
//     //     self.states.push(state);
//     //     self.rewards.push(reward);
//     //     self.actions.push(action);
//     //     self.dones.push(done);
//     //     self.log_probs.push(log_prob);
//     // }

//     pub fn submit_rollout(&mut self, experience: ExperienceStore) {  
//         // serialize data
//         let mut s = flexbuffers::FlexbufferSerializer::new();
//         experience.serialize(&mut s).unwrap();
//         let view = s.view();
//         // let view_dbg = view.to_vec();
//         // rpush experience
//         self.redis_con.rpush::<&str, Vec<u8>, ()>("exp_store", view.to_vec()).unwrap();
//     }
// }