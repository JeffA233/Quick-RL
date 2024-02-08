// use rlgym_sim_rs::GymManager;
use crate::gym_lib::GymManager;
// Vectorized version of the gym environment.
// use cpython::{buffer::PyBuffer, NoArgs, ObjectProtocol, PyObject, PyResult, Python};
use tch::{Tensor, Device, Kind};

#[derive(Debug)]
pub struct Step {
    pub obs: Tensor,
    pub reward: Tensor,
    pub is_done: Tensor,
}

pub struct VecGymEnv {
    // env: PyObject,
    env: GymManager,
    action_space: i64,
    observation_space: [i64; 2],
}

impl VecGymEnv {
    pub fn new(match_nums: Vec<usize>, gravity_nums: Vec<f32>, boost_nums: Vec<f32>, self_plays: Vec<bool>, tick_skip: usize, reward_file_name: String) -> VecGymEnv {
        let env = GymManager::new(match_nums, gravity_nums, boost_nums, self_plays, tick_skip, reward_file_name);
        let nprocesses = env.total_agents as i64;
        // FIXME: hardcoded for now
        // advancedobs in 1s is 107
        // advancedobs in 3s is 231
        let observation_space = 107;
        let action_space = 90;
        // let gil = Python::acquire_gil();
        // let py = gil.python();
        // let sys = py.import("sys")?;
        // let path = sys.get(py, "path")?;
        // let _ = path.call_method(py, "append", ("examples/reinforcement-learning",), None)?;
        // let gym = py.import("atari_wrappers")?;
        // let env = gym.call(py, "make", (name, img_dir, nprocesses), None)?;
        // let action_space = env.getattr(py, "action_space")?;
        // let action_space = action_space.getattr(py, "n")?.extract(py)?;
        // let observation_space = env.getattr(py, "observation_space")?;
        // let observation_space: Vec<i64> = observation_space.getattr(py, "shape")?.extract(py)?;
        // let observation_space =
        // [vec![nprocesses].as_slice(), observation_space.as_slice()].concat();
        let observation_space =
            [nprocesses, observation_space];
        VecGymEnv { env, action_space, observation_space }
    }

    pub fn reset(&self) -> Tensor {
        // let gil = Python::acquire_gil();
        // let py = gil.python();
        // let obs = self.env.call_method(py, "reset", NoArgs, None)?;
        // let obs = obs.call_method(py, "flatten", NoArgs, None)?;
        let obs_vecvec = self.env.reset();
        // let mut flattened_obs = Vec::new();
        // for obs in obs_vecvec {
        //     flattened_obs.extend(obs);
        // }
        // let obs = Tensor::from_slice(&flattened_obs);
        let obs = Tensor::from_slice2(&obs_vecvec);
        obs.view_(self.observation_space)
    }

    pub fn step(&mut self, actions: Vec<i64>, device: Device) -> Step {
        // let gil = Python::acquire_gil();
        // let py = gil.python();
        // let step = self.env.call_method(py, "step", (action,), None)?;
        // let obs = step.get_item(py, 0)?.call_method(py, "flatten", NoArgs, None)?;
        // let obs_buffer = PyBuffer::get(py, &obs)?;
        // let obs_vec: Vec<u8> = obs_buffer.to_vec(py)?;
        let mut actual_acts = Vec::new();
        for act in actions {
            actual_acts.push(vec![act as f32]);
        }

        self.env.step_async(actual_acts);
        let (obs_vec, rews, dones, infos, term_obs) = self.env.step_wait();
        // let dones_f32: Vec<f32> = dones.iter().map(|val| *val as usize as f32 ).collect();
        let (obs, reward, is_done) = if device == Device::Cpu {
            {
                (
                Tensor::from_slice2(&obs_vec).view_(self.observation_space),
                Tensor::from_slice(&rews),
                Tensor::from_slice(&dones)
                )
            }
        } else {
            {
                (
                    Tensor::from_slice2(&obs_vec).view_(self.observation_space).to_device_(device, Kind::Float, true, false),
                    Tensor::from_slice(&rews).to_device_(device, Kind::Float, true, false),
                    Tensor::from_slice(&dones).to_device_(device, Kind::Bool, true, false)
                )
            }

        };
        // let obs = Tensor::from_slice2(&obs_vec).view_(self.observation_space).pin_memory(Device::Cpu).to_device_(device, Kind::Float, true, false);
        // let reward = Tensor::from_slice(&rews).pin_memory(Device::Cpu).to_device_(device, Kind::Float, true, false);
        // let is_done = Tensor::from_slice(&dones).pin_memory(Device::Cpu).to_device_(device, Kind::Bool, true, false);
        Step { obs, reward, is_done }
    }

    pub fn action_space(&self) -> i64 {
        self.action_space
    }

    /// space is of size [[nprocs, observation_size]]
    pub fn observation_space(&self) -> &[i64] {
        &self.observation_space
    }
}