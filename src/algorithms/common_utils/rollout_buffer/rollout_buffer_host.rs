use std::time::Duration;

use redis::{Client, Connection};


pub struct ExpAdvHolder {
    pub s_states: Vec<Vec<f32>>,
    pub s_rewards: Vec<f32>,
    pub s_actions: Vec<f32>,
    pub dones_f: Vec<f32>,
    pub s_log_probs: Vec<f32>,
    pub advs: Vec<f32>,
    pub targ_vals: Vec<f32>,
}

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

    pub fn get_experience(&mut self, num_steps: usize, ) -> ExpAdvHolder {
        // TODO:
        // - blpop until we reach num_steps
        // - calculate GAE?

        // ExpAdvHolder {  }
        todo!()
    }
}