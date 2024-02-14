use std::time::Duration;

use redis::{Client, Connection};

use super::rollout_buffer_utils::ExperienceStore;


pub struct RolloutBufferWorker {
    // redis_url: String,
    redis_con: Connection,
}

impl RolloutBufferWorker {
    pub fn new(redis_url: String) -> Self {
        let redis_client = Client::open(redis_url).unwrap();
        let redis_con = redis_client.get_connection_with_timeout(Duration::from_secs(30)).unwrap();
        Self {
            // redis_url,
            redis_con
        }
    }

    pub fn submit_experience(&mut self, experience: ExperienceStore) {
        // TODO:
        // - rpush experience
        // - calculate GAE?
    }
}