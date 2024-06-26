use std::time::Duration;

use redis::{Client, Commands, Connection};
use serde::{Deserialize, Serialize};

use crate::algorithms::common_utils::rollout_buffer::rollout_buffer_utils::ExperienceStore;

use super::rollout_buffer_utils::RolloutDatabaseBackend;

pub struct RedisDatabaseBackend {
    redis_con: Connection,
}

impl RedisDatabaseBackend {
    pub fn new(redis_url: String) -> Self {
        let redis_client = Client::open(redis_url).unwrap();
        let redis_con = redis_client
            .get_connection_with_timeout(Duration::from_secs(30))
            .unwrap();
        Self { redis_con }
    }
}

impl RolloutDatabaseBackend for RedisDatabaseBackend {
    fn get_experience(&mut self, num_steps: usize, min_ver: i64) -> ExperienceStore {
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

        ExperienceStore {
            s_states: states,
            s_rewards: rewards,
            s_actions: actions,
            dones_f: dones,
            s_log_probs: log_probs,
            terminal_obs: term_obs,
            model_ver: 0,
        }
    }

    fn get_key_value_i64(&mut self, key: &str) -> Result<i64, Box<dyn std::error::Error>> {
        self.redis_con.get(key).map_err(|e| e.into())
    }

    fn get_key_value_bool(&mut self, key: &str) -> Result<bool, Box<dyn std::error::Error>> {
        self.redis_con.get(key).map_err(|e| e.into())
    }

    fn rpush(&mut self, key: &str, value: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        self.redis_con.rpush::<&str, &[u8], ()>(key, value)?;
        Ok(())
    }

    fn get_key_value_raw(&mut self, key: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        self.redis_con.get(key).map_err(|e| e.into())
    }

    fn set_key_value(
        &mut self,
        key: &str,
        value: impl Serialize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // let serialized_value = to_string(&value)?;
        let mut s = flexbuffers::FlexbufferSerializer::new();
        value.serialize(&mut s).unwrap();
        self.redis_con.set(key, s.view())?;
        Ok(())
    }

    fn set_key_value_raw(
        &mut self,
        key: &str,
        value: &[u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.redis_con.set(key, value)?;
        Ok(())
    }

    fn set_key_value_i64(
        &mut self,
        key: &str,
        value: i64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.redis_con.set(key, value)?;
        Ok(())
    }

    fn set_key_value_bool(
        &mut self,
        key: &str,
        value: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.redis_con.set(key, value)?;
        Ok(())
    }

    fn del(&mut self, key: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.redis_con.del(key)?;
        Ok(())
    }

    fn incr(&mut self, key: &str, increment: i64) -> Result<i64, Box<dyn std::error::Error>> {
        let result: i64 = self.redis_con.incr(key, increment)?;
        Ok(result)
    }
}
