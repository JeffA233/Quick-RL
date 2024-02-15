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

    pub fn get_experience(&mut self, num_steps: usize, ) -> ExperienceStore {
        let mut states = Vec::new();
        let mut rewards = Vec::new();
        let mut actions = Vec::new();
        let mut dones = Vec::new();
        let mut log_probs = Vec::new();

        while states.len() < num_steps {
            let exp_store: Vec<u8> = self.redis_con.blpop("exp_store", 0.).unwrap();
            let flex_read = flexbuffers::Reader::get_root(exp_store.as_slice()).unwrap();

            let exp_store = ExperienceStore::deserialize(flex_read).unwrap();

            states.extend(exp_store.s_states);
            rewards.extend(exp_store.s_rewards);
            actions.extend(exp_store.s_actions);
            dones.extend(exp_store.dones_f);
            log_probs.extend(exp_store.s_log_probs);
        }

        ExperienceStore { s_states: states, s_rewards: rewards, s_actions: actions, dones_f: dones, s_log_probs: log_probs }
    }
}