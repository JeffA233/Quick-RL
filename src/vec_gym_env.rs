// use rlgym_sim_rs::GymManager;
use crate::gym_lib::GymManager;
// Vectorized version of the gym environment.
use tch::{
    Device,
    // Kind,
    Tensor,
};

#[derive(Debug)]
pub struct Step {
    pub obs: Vec<Vec<f32>>,
    pub reward: Vec<f32>,
    pub is_done: Vec<bool>,
    // pub obs: Tensor,
    // pub reward: Tensor,
    // pub is_done: Tensor,
}

#[derive(Debug)]
pub struct EnvConfig {
    pub match_nums: Vec<usize>,
    pub gravity_nums: Vec<f32>,
    pub boost_nums: Vec<f32>,
    pub self_plays: Vec<bool>,
    pub tick_skip: usize,
    pub reward_file_name: String,
}

pub struct VecGymEnv {
    // env: PyObject,
    env: GymManager,
    action_space: i64,
    observation_space: [i64; 2],
}

impl VecGymEnv {
    pub fn new(
        match_nums: Vec<usize>,
        // gravity_nums: Vec<f32>,
        // boost_nums: Vec<f32>,
        self_plays: Vec<bool>,
        tick_skip: usize,
        reward_file_name: String,
    ) -> VecGymEnv {
        let env = GymManager::new(match_nums, self_plays, tick_skip, reward_file_name);
        let nprocesses = env.total_agents as i64;
        // FIXME: hardcoded for now
        // advancedobs in 1s is 107
        // advancedobs in 3s is 231
        let observation_space = 107;
        let action_space = 90;
        let observation_space = [nprocesses, observation_space];
        VecGymEnv {
            env,
            action_space,
            observation_space,
        }
    }

    pub fn reset(&self) -> Tensor {
        let obs_vecvec = self.env.reset();
        let obs = Tensor::from_slice2(&obs_vecvec);
        obs.view_(self.observation_space)
    }

    pub fn step(&mut self, actions: Vec<i64>, _device: Device) -> Step {
        let mut actual_acts = Vec::new();
        for act in actions {
            actual_acts.push(vec![act as f32]);
        }

        self.env.step_async(actual_acts);
        let (obs, reward, is_done, _infos, _term_obs) = self.env.step_wait();
        // let dones_f32: Vec<f32> = dones.iter().map(|val| *val as usize as f32 ).collect();
        // let (obs, reward, is_done) = if device == Device::Cpu {
        //     {
        //         (
        //         Tensor::from_slice2(&obs).view_(self.observation_space),
        //         Tensor::from_slice(&reward),
        //         Tensor::from_slice(&is_done)
        //         )
        //     }
        // } else {
        //     {
        //         (
        //             Tensor::from_slice2(&obs).view_(self.observation_space).to_device_(device, Kind::Float, true, false),
        //             Tensor::from_slice(&reward).to_device_(device, Kind::Float, true, false),
        //             Tensor::from_slice(&is_done).to_device_(device, Kind::Bool, true, false)
        //         )
        //     }

        // };
        // let obs = Tensor::from_slice2(&obs_vec).view_(self.observation_space).pin_memory(Device::Cpu).to_device_(device, Kind::Float, true, false);
        // let reward = Tensor::from_slice(&rews).pin_memory(Device::Cpu).to_device_(device, Kind::Float, true, false);
        // let is_done = Tensor::from_slice(&dones).pin_memory(Device::Cpu).to_device_(device, Kind::Bool, true, false);
        // Step { obs, reward, is_done }
        Step {
            obs,
            reward,
            is_done,
        }
    }

    pub fn action_space(&self) -> i64 {
        self.action_space
    }

    /// space is of size [[nprocs, observation_size]]
    pub fn observation_space(&self) -> Vec<i64> {
        self.observation_space.to_vec()
    }
}
