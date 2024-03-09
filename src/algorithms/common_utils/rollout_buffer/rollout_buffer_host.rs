// use std::time::Duration;

// use redis::{Client, Commands, Connection};
// use serde::{Deserialize, Serialize};

// use super::rollout_buffer_utils::ExperienceStore;

// pub trait RolloutHostBackend {
//     fn get_experience(&mut self, num_steps: usize, min_ver: i64) -> ExperienceStore;
//     fn set_key_value(
//         &mut self,
//         key: &str,
//         value: impl Serialize,
//     ) -> Result<(), Box<dyn std::error::Error>>;
//     fn set_key_value_raw(
//         &mut self,
//         key: &str,
//         value: &[u8],
//     ) -> Result<(), Box<dyn std::error::Error>>;
//     fn set_key_value_i64(
//         &mut self,
//         key: &str,
//         value: i64,
//     ) -> Result<(), Box<dyn std::error::Error>>;
//     fn set_key_value_bool(
//         &mut self,
//         key: &str,
//         value: bool,
//     ) -> Result<(), Box<dyn std::error::Error>>;
//     fn del(&mut self, key: &str) -> Result<(), Box<dyn std::error::Error>>;
//     fn incr(&mut self, key: &str, increment: i64) -> Result<i64, Box<dyn std::error::Error>>;
// }
