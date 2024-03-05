use serde::Deserialize;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Deserialize)]
pub struct Configuration {
    pub hyperparameters: Hyperparameters,
    pub device: String,
    pub reward_file_full_path: String,
    pub n_env: i64,
    pub tick_skip: usize,
    pub n_stack: i64,
    pub gamemodes: Gamemodes,
    pub redis: Redis,
    pub network: Network,
}

#[derive(Deserialize)]
pub struct Hyperparameters {
    pub entropy_coef: f64,
    pub clip_range: f64,
    pub grad_clip: f64,
    pub lr: f64,
    pub gamma: f64,
    pub steps_per_rollout: i64,
    pub updates: i64,
    pub max_model_age: i64,
    pub buffersize: usize,
    pub optim_epochs: i64,
}

#[derive(Deserialize)]
pub struct Gamemodes {
    pub num_1s: usize,
    pub num_1s_gravboost: usize,
    pub num_1s_selfplay: usize,
    pub num_2s: usize,
    pub num_2s_gravboost: usize,
    pub num_2s_selfplay: usize,
    pub num_3s: usize,
    pub num_3s_gravboost: usize,
    pub num_3s_selfplay: usize,
}

fn default_as_str_0() -> String {
    "0".to_string()
}

fn default_as_str_empty() -> String {
    "".to_string()
}

fn default_localhost() -> String {
    "localhost".to_string()
}


fn default_default() -> String {
    "default".to_string()
}

#[derive(Deserialize)]
pub struct Redis {
    #[serde(default = "default_localhost")]
    pub ipaddress: String,
    #[serde(default = "default_default")]
    pub username: String,
    #[serde(default = "default_as_str_empty")]
    pub password_env_var: String,
    #[serde(default = "default_as_str_0")]
    pub dbnum: String,
}

#[derive(Deserialize)]
pub struct Network {
    pub custom_shape: bool,
    pub act_func: String,
    pub actor: LayerConfig,
    pub critic: LayerConfig,
    pub custom_actor: CustomLayerConfig,
    pub custom_critic: CustomLayerConfig,
}

#[derive(Deserialize)]
pub struct LayerConfig {
    pub num_layers: usize,
    pub layer_size: i64,
}

#[derive(Deserialize)]
pub struct CustomLayerConfig {
    pub layer_vec: Vec<i64>
}

impl Configuration {
    pub fn load_configuration(config_file: &Path) -> Result<Configuration, serde_json::Error> {
        let mut file = match File::open(config_file) {
            Ok(file) => file,
            Err(error) => {
                panic!("Error opening file {}: {}", config_file.display(), error);
            }
        };
        let mut contents = String::new();
        match file.read_to_string(&mut contents) {
            Ok(_) => (), // Reading was successful
            Err(error) => {
                panic!("Error reading contents of {}: {}", config_file.display(), error);
            }
        };
        serde_json::from_str(&contents)
    }
}
