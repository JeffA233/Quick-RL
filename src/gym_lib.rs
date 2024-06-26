// NOTE: This is originally just a Python wrapper around rlgym-sim-rs with the possibility of doing vectorized envs across multiple threads.
// It has been converted to work solely in Rust. Taken from personal bot project (Matrix) using RocketSim/rlgym-sim-rs -Jeff

use rlgym_sim_gym::{
    envs::game_match::GameConfig, make, obs_builders, AdvancedObs, MakeConfig, RenderConfig,
};

use crossbeam_channel::{bounded, Receiver, Sender};
use itertools::izip;
use rocketsim_rs;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::{
    collections::HashMap,
    iter::zip,
    path::PathBuf,
    thread::{self, JoinHandle},
    time::Duration,
};

use rlgym_sim_gym::Gym;

use crate::gym_funcs::custom_conditions::CombinedTerminalConditions;
use crate::gym_funcs::custom_rewards::GatherBoostRewardBasic;
use crate::gym_funcs::custom_state_setters::custom_state_setters;
use crate::gym_funcs::necto_parser_2::NectoAction;
use obs_builders::obs_builder::ObsBuilder;

/// A Python module implemented in Rust.
// #[pymodule]
// pub fn rlgym_sim_rs(_py: Python, m: &PyModule) -> PyResult<()> {
//     m.add_class::<GymWrapper>()?;
//     m.add_class::<GymManager>()?;
//     m.add_class::<GymWrapperRender>()?;
//     Ok(())
// }

/// Wrapper for a singular gym to be used in Python as a module
// #[pyclass(unsendable)]
pub struct GymWrapper {
    gym: Gym,
}

// #[pymethods]
impl GymWrapper {
    // #[new]
    /// create the gym wrapper to be used (team_size: i32, tick_skip: usize)
    pub fn new(team_size: usize, tick_skip: usize, seed: Option<u64>) -> Self {
        let term_cond = Box::new(CombinedTerminalConditions::new(tick_skip));
        let reward_fn = Box::new(GatherBoostRewardBasic::new());
        let obs_build = Box::new(AdvancedObs::new());
        let obs_build_vec: Vec<Box<dyn ObsBuilder>> = vec![obs_build];
        let act_parse = Box::new(NectoAction::new());
        let state_set = Box::new(custom_state_setters(team_size, Some(0)));
        let game_config = GameConfig {
            tick_skip,
            team_size,
            ..Default::default()
        };
        let make_config = MakeConfig {
            game_config,
            terminal_condition: term_cond,
            reward_fn,
            obs_builder: obs_build_vec,
            use_single_obs: true,
            action_parser: act_parse,
            state_setter: state_set,
        };
        let gym = make::make(make_config, None);
        GymWrapper { gym }
    }

    pub fn reset(&mut self, seed: Option<u64>) -> Vec<Vec<f32>> {
        self.gym.reset(Some(false), Some(0))
    }

    pub fn step(
        &mut self,
        actions: Vec<Vec<f32>>,
    ) -> (Vec<Vec<f32>>, Vec<f32>, bool, HashMap<String, f32>) {
        self.gym.step(actions)
    }

    // pub fn close(&mut self) -> PyResult<()> {
    //     self.gym.close();
    //     Ok(())
    // }
}

/// Wrapper for a singular gym to be used in Python as a module
// #[pyclass(unsendable)]
pub struct GymWrapperRender {
    gym: Gym,
}

// #[pymethods]
impl GymWrapperRender {
    // #[new]
    /// create the gym wrapper to be used (team_size: i32, tick_skip: usize)
    pub fn new(
        team_size: usize,
        gravity: f32,
        boost: f32,
        self_play: bool,
        tick_skip: usize,
        render: bool,
        render_speed: f32,
        seed: Option<u64>,
    ) -> Self {
        let term_cond = Box::new(CombinedTerminalConditions::new(tick_skip));
        let reward_fn = Box::new(GatherBoostRewardBasic::new());
        // let obs_build = Box::new(AdvancedObsPadderStacker3::new(None, Some(0)));
        let obs_build = Box::new(AdvancedObs::new());
        let obs_build_vec: Vec<Box<dyn ObsBuilder>> = vec![obs_build];
        let act_parse = Box::new(NectoAction::new());
        let state_set = Box::new(custom_state_setters(team_size, Some(0)));
        let game_config = GameConfig {
            gravity,
            boost_consumption: boost,
            team_size,
            tick_skip,
            spawn_opponents: self_play,
        };
        let make_config = MakeConfig {
            game_config,
            terminal_condition: term_cond,
            reward_fn,
            obs_builder: obs_build_vec,
            use_single_obs: true,
            action_parser: act_parse,
            state_setter: state_set,
        };
        let render_config = RenderConfig {
            render,
            update_rate: render_speed,
        };
        let gym = make::make(make_config, Some(render_config));
        Self { gym }
    }

    pub fn reset(&mut self, seed: Option<u64>) -> Vec<Vec<f32>> {
        self.gym.reset(Some(false), seed)
    }

    pub fn step(
        &mut self,
        actions: Vec<Vec<f32>>,
    ) -> (Vec<Vec<f32>>, Vec<f32>, bool, HashMap<String, f32>) {
        self.gym.step(actions)
    }

    pub fn close(&mut self) {
        self.gym.close_renderer();
    }
}
// end of Python RLGym env
// -------------------------------------------------------------------------------------
// gym wrapper trait so that this can be a crate and be provided reward fns and etc.

// pub trait GymWrapperTrait {
//     fn new(
//         team_size: i32,
//         gravity: f32,
//         boost: f32,
//         self_play: bool,
//         tick_skip: usize,
//         term_cond: Box<dyn TerminalCondition>,
//         reward_fn: Box<dyn RewardFn>,
//         obs_builder_vec: Vec<Box<dyn ObsBuilder + Send>>,
//         action_parser: Box<dyn ActionParser>,
//         state_setter: Box<dyn StateSetter>
//     ) -> Self;
//     fn reset(&mut self) -> Vec<Vec<f32>>;
//     fn step(&mut self, actions: Vec<Vec<f32>>) -> (Vec<Vec<f32>>, Vec<f32>, bool, HashMap<String, f32>);
// }

// end of gym wrapper trait
// -------------------------------------------------------------------------------------
// start of RLGym multiple-instance manager for Python

/// wrapper that is used internally (in Rust) around the gym
pub struct GymWrapperRust {
    gym: Gym,
}

impl GymWrapperRust {
    /// create the gym wrapper to be used (team_size: i32, tick_skip: usize)
    pub fn new(
        team_size: usize,
        gravity: f32,
        boost: f32,
        self_play: bool,
        tick_skip: usize,
        sender: Sender<Vec<f32>>,
    ) -> Self {
        let term_cond = Box::new(CombinedTerminalConditions::new(tick_skip));
        let reward_fn = Box::new(GatherBoostRewardBasic::new());
        // let mut obs_build_vec = Vec::<Box<dyn ObsBuilder>>::new();
        // for _ in 0..team_size * 2 {
        //     // obs_build_vec.push(Box::new(AdvancedObsPadderStacker3::new(None, Some(0))));
        //     obs_build_vec.push(Box::new(AdvancedObs::new()));
        // }
        let obs_build_vec: Vec<Box<dyn ObsBuilder>> = vec![Box::new(AdvancedObs::new())];
        // let obs_build = Box::new(AdvancedObsPadderStacker::new(None, Some(5)));
        let act_parse = Box::new(NectoAction::new());
        let state_set = Box::new(custom_state_setters(team_size, Some(0)));
        let game_config = GameConfig {
            gravity,
            boost_consumption: boost,
            team_size,
            tick_skip,
            spawn_opponents: self_play,
        };
        let make_config = MakeConfig {
            game_config,
            terminal_condition: term_cond,
            reward_fn,
            obs_builder: obs_build_vec,
            use_single_obs: true,
            action_parser: act_parse,
            state_setter: state_set,
        };
        let gym = make::make(make_config, None);
        // let gym = make::make(
        //     Some(100000.),
        //     Some(tick_skip),
        //     Some(self_play),
        //     Some(team_size),
        //     Some(gravity),
        //     Some(boost),
        //     term_cond,
        //     reward_fn,
        //     obs_build_vec,
        //     act_parse,
        //     state_set,
        // );
        GymWrapperRust { gym }
    }

    pub fn reset(&mut self) -> Vec<Vec<f32>> {
        self.gym.reset(Some(false), Some(0))
    }

    pub fn step(
        &mut self,
        actions: Vec<Vec<f32>>,
    ) -> (Vec<Vec<f32>>, Vec<f32>, bool, HashMap<String, f32>) {
        self.gym.step(actions)
    }

    // pub fn close(&mut self) {
    //     self.gym.close();
    // }
}

// RLGym wrapper for Rust (mostly used to preset the gym options)
// -------------------------------------------------------------------------------------
// RLGym Manager in Rust for Python interface

/// manager for multiple instances of the gym done in Rust for Python use
// #[pyclass]
pub struct GymManager {
    // #[pyo3(get)]
    waiting: bool,
    // threads: Vec<JoinHandle<()>>,
    sends: Vec<Sender<ManagerPacket>>,
    recvs: Vec<Receiver<WorkerPacket>>,
    n_agents_per_env: Vec<usize>,
    pub total_agents: usize,
}

/// packet that comes from the manager
pub enum ManagerPacket {
    Step { actions: Vec<Vec<f32>> },
    Reset,
    Close,
}

/// packet that comes from the worker
pub enum WorkerPacket {
    StepRet {
        obs: Vec<Vec<f32>>,
        reward: Vec<f32>,
        done: bool,
        info: HashMap<String, f32>,
    },
    StepRetDone {
        obs: Vec<Vec<f32>>,
        reward: Vec<f32>,
        done: bool,
        info: HashMap<String, f32>,
        terminal_obs: Vec<Vec<f32>>,
    },
    ResetRet {
        obs: Vec<Vec<f32>>,
    },
    InitReturn,
}

/// Python method that wraps all of the threaded gym instances and collects the data from each into the correct format to be pushed back to Python
// #[pymethods]
impl GymManager {
    // #[new]
    pub fn new(
        match_nums: Vec<usize>,
        self_plays: Vec<bool>,
        tick_skip: usize,
        reward_file_full_path: String,
    ) -> Self {
        rocketsim_rs::init(None);
        let mut recv_vec = Vec::<Receiver<WorkerPacket>>::new();
        let mut send_vec = Vec::<Sender<ManagerPacket>>::new();
        let mut thrd_vec = Vec::<JoinHandle<()>>::new();
        // let mut curr_id = 0;

        let (reward_send, reward_recv) = bounded(20000);
        let reward_path = Path::new(&reward_file_full_path).to_owned();
        // let reward_thrd = thread::spawn(move || file_put_worker(reward_recv, reward_path));
        thread::spawn(move || file_put_worker(reward_recv, reward_path));

        // redo agent numbers for self-play case, need to redo to just be agents on one team instead of for whole match
        let mut corrected_match_nums = Vec::<usize>::new();

        for (match_num, self_play) in match_nums.iter().zip(self_plays.iter()) {
            if *self_play {
                corrected_match_nums.push(*match_num * 2);
            } else {
                corrected_match_nums.push(*match_num);
            }
        }

        for (match_num, self_play) in izip!(match_nums, self_plays) {
            let mut retry_loop = true;
            let mut num_retries = 0;
            // try to loop until the game successfully launches
            while retry_loop {
                let reward_send_local = reward_send.clone();
                let send_local: Sender<ManagerPacket>;
                let rx: Receiver<ManagerPacket>;
                let tx: Sender<WorkerPacket>;
                let recv_local: Receiver<WorkerPacket>;
                (send_local, rx) = bounded(1);
                (tx, recv_local) = bounded(1);
                let thrd1 = thread::spawn(move || {
                    worker(
                        match_num,
                        1.,
                        1.,
                        self_play,
                        tick_skip,
                        tx,
                        rx,
                        reward_send_local,
                    )
                });
                // curr_id += 1;

                // wait for worker to send back a packet or if it never does then restart loop to try again
                let out = recv_local.recv_timeout(Duration::new(60, 0));

                match out {
                    Ok(packet) => packet,
                    Err(err) => {
                        println!("recv timed out in new func: {err}");
                        num_retries += 1;
                        if num_retries >= 10 {
                            // In case the gym has a legitimate and repeatable error, we don't want to let it repeat for infinity trying to work.
                            // Not really necessary for RocketSim as a single error should be considered problematic unlike Rocket League where it may
                            // just simply crash for no reason.
                            panic!("hit repeat limit when trying to create gyms: {num_retries}");
                        } else {
                            continue;
                        }
                    }
                };

                // gather all of the local channels and threads for later use (if game launches are successful)
                recv_vec.push(recv_local);
                send_vec.push(send_local);
                thrd_vec.push(thrd1);
                retry_loop = false;
            }
        }

        GymManager {
            waiting: false,
            // threads: thrd_vec,
            sends: send_vec,
            recvs: recv_vec,
            n_agents_per_env: corrected_match_nums.clone(),
            total_agents: corrected_match_nums.iter().sum::<usize>(),
        }
    }

    pub fn reset(&self) -> Vec<Vec<f32>> {
        for sender in &self.sends {
            sender.send(ManagerPacket::Reset).unwrap();
        }
        // self.sends.par_iter().for_each(|send| send.send(ManagerPacket::Reset).unwrap());

        // flat obs means that the obs should be of shape [num_envs, obs_size] (except this is a Vec so it's not a "shape" but the length)
        let mut flat_obs = Vec::<Vec<f32>>::with_capacity(self.total_agents);
        for receiver in &self.recvs {
            let data = receiver.recv().unwrap();
            let obs = match data {
                WorkerPacket::ResetRet { obs } => obs,
                _ => panic!("ResetRet was not returned from Reset command given"),
            };
            for internal_vec in obs {
                flat_obs.push(internal_vec);
            }
        }
        // let flat_obs_numpy = PyArray::from_vec2(py, &flat_obs).unwrap().to_owned();
        // return Ok(flat_obs)
        // Ok(flat_obs_numpy)
        flat_obs
    }

    pub fn step_async(&mut self, actions: Vec<Vec<f32>>) {
        let mut i: usize = 0;
        for (sender, agent_num) in zip(&self.sends, &self.n_agents_per_env) {
            let acts = actions[i..i + *agent_num].to_vec();
            let out = sender.send(ManagerPacket::Step { actions: acts });
            match out {
                Ok(val) => val,
                Err(err) => {
                    // return Err(pyo3::exceptions::PyBrokenPipeError::new_err(format!("Send error in step_async(): {}", err)))
                    panic!("Send error in step_async(): {}", err)
                }
            }
            i += *agent_num
        }
        self.waiting = true;
        // Ok(())
    }

    pub fn step_wait(
        &mut self,
        // py: Python<'_>,
    ) -> (
        Vec<Vec<f32>>,
        Vec<f32>,
        Vec<bool>,
        Vec<HashMap<String, f32>>,
        Vec<Option<Vec<Vec<f32>>>>,
    ) {
        let mut flat_obs = Vec::<Vec<f32>>::with_capacity(self.total_agents);
        let mut flat_rewards = Vec::<f32>::with_capacity(self.total_agents);
        let mut flat_dones = Vec::<bool>::with_capacity(self.total_agents);
        let mut flat_infos = Vec::<HashMap<String, f32>>::with_capacity(self.total_agents);
        let mut flat_term_obs = Vec::<Option<Vec<Vec<f32>>>>::with_capacity(self.total_agents);

        for (receiver, n_agents) in zip(&self.recvs, &self.n_agents_per_env) {
            let data = receiver.recv().unwrap();

            let (obs, rew, done, info, terminal_obs) = match data {
                WorkerPacket::StepRet {
                    obs,
                    reward,
                    done,
                    info,
                } => (obs, reward, done, info, vec![None; *n_agents]),
                WorkerPacket::StepRetDone {
                    obs,
                    reward,
                    done,
                    info,
                    terminal_obs,
                } => (obs, reward, done, info, vec![Some(terminal_obs)]),
                _ => panic!("StepRet was not returned from Step command given"),
            };
            // same as above in reset fn and for rewards it will be a vec of f32 to be "flat" and so on
            flat_obs.extend(obs);

            flat_rewards.extend(rew);

            // since PyO3 cannot currently use HashMaps with enums we must push this outside of Rust into Python
            // match terminal_obs {
            //     Some(obs) => flat_term_obs.extend(vec![Some(obs); *n_agents]),
            //     None => flat_term_obs.extend(vec![None; *n_agents]),
            // };
            flat_term_obs.extend(terminal_obs);
            // since the env will emit done and info as the same for every agent in the match, we just multiply them to fill the number of agents
            flat_dones.extend(vec![done; *n_agents]);
            flat_infos.extend(vec![info; *n_agents]);
        }
        // let flat_obs_numpy = PyArray::from_vec2(py, &flat_obs).unwrap().to_owned();
        self.waiting = false;

        (
            flat_obs,
            flat_rewards,
            flat_dones,
            flat_infos,
            flat_term_obs,
        )
    }

    pub fn close(&mut self) {
        for sender in &self.sends {
            sender.send(ManagerPacket::Close).unwrap();
        }
    }
}

/// a sort of wrapper fn for doing the individual steps for an instance of the gym, should be launched on its own individual thread
pub fn worker(
    team_num: usize,
    gravity: f32,
    boost: f32,
    self_play: bool,
    tick_skip: usize,
    send_chan: Sender<WorkerPacket>,
    rec_chan: Receiver<ManagerPacket>,
    reward_sender: Sender<Vec<f32>>,
) {
    // launches env and then sends the reset action to a new thread since receiving a message from the plugin will be blocking,
    // waits for x seconds for thread to return the env if it is a success else tries to force close the pipe and
    // make the gym crash (which should terminate the game)
    let mut env = GymWrapperRust::new(
        team_num,
        gravity,
        boost,
        self_play,
        tick_skip,
        reward_sender,
    );
    send_chan.send(WorkerPacket::InitReturn).unwrap();

    loop {
        // simple loop that tries to recv for as long as the Manager channel is not hung up waiting for commands from the Manager
        let mut obs: Vec<Vec<f32>>;
        let reward: Vec<f32>;
        let done: bool;
        let info: HashMap<String, f32>;
        let recv_data = rec_chan.recv();
        let cmd = match recv_data {
            Ok(out) => out,
            Err(err) => {
                println!("recv err in worker: {err}");
                break;
            }
        };
        match cmd {
            ManagerPacket::Step { actions } => {
                (obs, reward, done, info) = env.step(actions);
                let out;
                // check if match is done, unfortunately we must send this to Python somehow because HashMaps with enums
                // (for multiple-type HashMaps) cannot be translated into Python
                if done {
                    let terminal_obs = obs;
                    obs = env.reset();
                    out = send_chan.send(WorkerPacket::StepRetDone {
                        obs,
                        reward,
                        done,
                        info,
                        terminal_obs,
                    });
                } else {
                    out = send_chan.send(WorkerPacket::StepRet {
                        obs,
                        reward,
                        done,
                        info,
                    });
                }
                match out {
                    Ok(res) => res,
                    Err(err) => {
                        println!("send err in worker: {err}");
                        break;
                    }
                }
            }
            ManagerPacket::Close => break,
            ManagerPacket::Reset => {
                obs = env.reset();
                // send_chan.send(WorkerPacket::ResetRet { obs }).unwrap();
                let out = send_chan.send(WorkerPacket::ResetRet { obs });
                match out {
                    Ok(res) => res,
                    Err(err) => {
                        println!("send err in worker: {err}");
                        break;
                    }
                }
            }
        };
    }
}

/// write side of the logger for SB3, receives a vec of individual reward fn values
fn file_put_worker(receiver: Receiver<Vec<f32>>, reward_file: PathBuf) {
    loop {
        let out = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(reward_file.as_path());

        let file = match out {
            Err(out) => {
                println!("file error: {out}");
                // half a second
                thread::sleep(Duration::new(0, 500000000));
                // if out.kind() == PermissionDenied {continue} else {continue};
                continue;
            }
            Ok(_file) => _file,
        };
        let mut buf = BufWriter::new(&file);

        let mut i = 0;
        loop {
            let recv_data = receiver.recv();
            let returns_local = match recv_data {
                Ok(data) => data,
                Err(err) => {
                    println!(
                        "recv err in file_put_worker using reward_file {}: {}",
                        reward_file.display(),
                        err
                    );
                    break;
                }
            };

            // format rewards string to be readable in Python
            let mut string = String::new();
            string += "[";
            for ret in returns_local.iter().take(returns_local.len() - 1) {
                string = string + &format!("{}, ", ret)
            }
            string = string + &format!("{}]", returns_local[returns_local.len() - 1]);
            writeln!(&mut buf, "{}", string).unwrap();

            i += 1;

            if receiver.is_empty() || i > 1000 {
                let out = buf.flush();
                match out {
                    Ok(out) => out,
                    Err(err) => println!("buf.flush in logger failed with error: {err}"),
                };
                i = 0;
                // break;
            }
        }
        println!("logger worker is exiting");
        break;
    }
}

// this will be used to consume data from one or multiple gym instances to update
// fn vis_state_logger(receiver: Receiver<(GameState_rocketsim, Vec<(u32, CarControls)>)>) {
//     let max_len = 10;
//     let mut receiver_holder = VecDeque::with_capacity(max_len + 1);
//     loop {
//         if !receiver.is_empty() {
//             let data_op = receiver.recv();
//             match data_op {
//                 Ok(data) => {
//                     receiver_holder.push_front(data);
//                     if receiver_holder.len() > max_len {
//                         receiver_holder.truncate(max_len);
//                     }
//                 }
//                 Err(e) => println!("error receiving data in state logger: {e}"),
//             };
//         }
//     }
// }
