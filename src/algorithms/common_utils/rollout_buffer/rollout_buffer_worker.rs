use std::time::Duration;

use crossbeam_channel::Receiver;
use redis::{Client, Commands, Connection};
use serde::Serialize;

use super::rollout_buffer_utils::ExperienceStore;

pub trait RolloutWorkerBackend {
    fn get_key_value_i64(&mut self, key: &str) ->  Result<i64, Box<dyn std::error::Error>>;
    fn get_key_value_bool(&mut self, key: &str) ->  Result<bool, Box<dyn std::error::Error>>;
    fn get_key_value_raw(&mut self, key: &str) ->  Result<Vec<u8>, Box<dyn std::error::Error>>;
    fn rpush(&mut self, key: &str, value: &[u8]) -> Result<(), Box<dyn std::error::Error>>;
}

// pub trait RolloutWorkerSimple {
//     fn get_key_value_bool<K: AsRef<str>>(&mut self, key: K) ->  Result<bool, Box<dyn std::error::Error>>;
// }

pub struct RolloutWorker<T: RolloutWorkerBackend> {
    backend: T,
    states: Vec<Vec<f32>>,
    rewards: Vec<f32>,
    actions: Vec<f32>,
    dones: Vec<f32>,
    log_probs: Vec<f32>,
    model_version: i64,
}

impl<T: RolloutWorkerBackend> RolloutWorker<T> {
    pub fn new(backend_fn: &(dyn Fn() -> T + Send + 'static), obs_space: i64, nsteps: i64) -> Self {
        let mut states = Vec::with_capacity(nsteps as usize);
        states.push(vec![0.; obs_space as usize]);
        let backend: T = backend_fn();
        Self {
            backend,
            states,
            rewards: Vec::with_capacity(nsteps as usize),
            actions: Vec::with_capacity(nsteps as usize),
            dones: Vec::with_capacity(nsteps as usize),
            log_probs: Vec::with_capacity(nsteps as usize),
            model_version: 0,
        }
    }
}

impl<T: RolloutWorkerBackend> RolloutWorker<T> {
    /// note here that the state is actually the previous state t+0 from the gym and not the current one which is t+1
    fn push_experience(&mut self, state: Vec<f32>, reward: f32, action: f32, done: f32, log_prob: f32, model_ver: i64) -> bool {
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
            self.submit_rollout(ExperienceStore { 
                s_states: self.states.clone(), 
                s_rewards: self.rewards.clone(), 
                s_actions: self.actions.clone(), 
                dones_f: self.dones.clone(), 
                s_log_probs: self.log_probs.clone(), 
                terminal_obs: last_state.clone(), 
                model_ver: self.model_version,
            });
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

            self.model_version = model_ver;

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
        self.backend.rpush("exp_store", view).unwrap();

        // self.states = Vec::new();
        // self.rewards = Vec::new();
        // self.actions = Vec::new();
        // self.dones = Vec::new();
        // self.log_probs = Vec::new();
    }
}

pub struct RedisWorkerBackend {
    redis_con: Connection,
}

impl RedisWorkerBackend {
    pub fn new(redis_url: String) -> Self {
        let redis_client = Client::open(redis_url).unwrap();
        let redis_con = redis_client.get_connection_with_timeout(Duration::from_secs(30)).unwrap();
        Self { redis_con }
    }
}

impl RolloutWorkerBackend for RedisWorkerBackend{

    fn get_key_value_i64(&mut self, key: &str) -> Result<i64, Box<dyn std::error::Error>> {
        self.redis_con.get(key)
            .map_err(|e| e.into())
    }

    fn get_key_value_bool(&mut self, key: &str) -> Result<bool, Box<dyn std::error::Error>> {
        self.redis_con.get(key)
            .map_err(|e| e.into())
    }

    fn rpush(&mut self, key: &str, value: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        self.redis_con.rpush::<&str, &[u8], ()>(key, value)?;
        Ok(())
    }

    fn get_key_value_raw(&mut self, key: &str) ->  Result<Vec<u8>, Box<dyn std::error::Error>> {
        self.redis_con.get(key)
        .map_err(|e| e.into())
    }
}

pub struct StepStore {
    pub obs: Vec<Vec<f32>>,
    pub action: Vec<f32>,
    pub reward: Vec<f32>,
    pub done: Vec<f32>,
    pub log_prob: Vec<f32>,
    pub model_ver: i64,
    // info: HashMap<String, f32>,
}

pub fn buffer_worker<T: RolloutWorkerBackend, F: Fn() -> T + Send + 'static>(
    rec_chan: Receiver<StepStore>,
    backend_factory: F,
    obs_space: i64,
    nsteps: i64,
    nprocs: usize,
) {
    // let rollout_worker = RolloutBufferWorker::new(redis_url, obs_space, nsteps);
    let mut rollout_bufs = Vec::new();
    
    // let rollout_worker = RolloutWorker::new(redis_backend, obs_space, nsteps);
    for _i in 0..nprocs {
        // let backend_box = backend_factory();
        // dbg!(backend.);
        rollout_bufs.push(RolloutWorker::new(&backend_factory, obs_space, nsteps));
    }

    loop {
        let recv_data = rec_chan.recv();
        let step_store = match recv_data {
            Ok(out) => out,
            Err(err) => {
                // NOTE: remove for now since we're just running the function aka the channel wil be disconnected anyways
                // TODO: handle this error
                println!("recv err in experience buf worker: {err}");
                break;
            }
        };
        // rollout_worker.push_experience(step_store.obs, step_store.reward, step_store.action, step_store.done, step_store.log_prob);
        for (i, buf) in rollout_bufs.iter_mut().enumerate() {
            buf.push_experience(step_store.obs[i].clone(), step_store.reward[i], step_store.action[i], step_store.done[i], step_store.log_prob[i], step_store.model_ver);
        }
        
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