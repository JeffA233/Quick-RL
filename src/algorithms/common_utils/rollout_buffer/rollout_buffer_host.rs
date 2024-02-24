use std::time::Duration;

use redis::{Client, Commands, Connection};
use serde::Deserialize;

use super::rollout_buffer_utils::ExperienceStore;


pub struct RolloutBufferHost {
    // redis_url: String,
    redis_con: Connection,
}

impl RolloutBufferHost {
    pub fn new(redis_url: String) -> Self {
        let redis_client = Client::open(redis_url).unwrap();
        let redis_con = redis_client.get_connection_with_timeout(Duration::from_secs(30)).unwrap();
        Self {
            // redis_url,
            redis_con
        }
    }

    pub fn get_experience(&mut self, num_steps: usize, min_ver: i64) -> ExperienceStore {
        let mut states = Vec::new();
        let mut rewards = Vec::new();
        let mut actions = Vec::new();
        let mut dones = Vec::new();
        let mut log_probs = Vec::new();
        let mut term_obs = Vec::new();

        let mut discarded_stores = 0;

        while rewards.len() < num_steps {
            // blpop can get multiple keys so it will return the key-name followed by the actual data for each key input
            // [["n, a, m, e"], ["d, a, t, a"]]
            let exp_store_bytes: Vec<Vec<u8>> = self.redis_con.blpop("exp_store", 0.).unwrap();
            // let exp_store: Vec<u8> = self.redis_con.blpop("exp_store", 0.).unwrap();
            // let exp_store_str: String = self.redis_con.blpop("exp_store", 0.).unwrap();
            // let exp_store = exp_store_str.as_bytes();
            let flex_read = flexbuffers::Reader::get_root(exp_store_bytes[1].as_slice()).unwrap();

            let exp_store = ExperienceStore::deserialize(flex_read).unwrap();

            if exp_store.model_ver > min_ver {
                states.extend(exp_store.s_states);
                rewards.extend(exp_store.s_rewards);
                actions.extend(exp_store.s_actions);
                dones.extend(exp_store.dones_f);
                log_probs.extend(exp_store.s_log_probs);
    
                term_obs = exp_store.terminal_obs;
            } else {
                discarded_stores += 1;
                // println!("discarded steps, model ver was {} and min version was: {}", exp_store.model_ver, min_ver);
            }
        }

        println!("discarded {} rollouts", discarded_stores);

        ExperienceStore { s_states: states, s_rewards: rewards, s_actions: actions, dones_f: dones, s_log_probs: log_probs, terminal_obs: term_obs, model_ver: 0, }
    }
}