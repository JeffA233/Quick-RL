# Quick-RL
Test project to validate if [tch-rs](https://github.com/LaurentMazare/tch-rs) and Rust are viable to be used for the full training side of reinforcement learning (though specifically Proximal Policy Optimization). Minor inspiration of structural layout from [rlgym-ppo](https://github.com/AechPro/rlgym-ppo). It has fully asynchronous workers that can be distributed if desired but is limited to one learner instance. 

Currently it is setup to use Redis as a means to communicate between worker(s) and learner. There is the ability to change to another networking backend if desired by just reimplementing the [RolloutDatabaseBackend](https://github.com/JeffA233/Quick-RL/blob/master/src/algorithms/common_utils/rollout_buffer/rollout_buffer_utils.rs#L49) trait. It also uses a [config file](https://github.com/JeffA233/Quick-RL/blob/master/src/config.json) to allow for rapid reconfiguration if necessary.

## Rough Examples
See the [bin](https://github.com/JeffA233/Quick-RL/tree/master/src/bin) folder for the worker and learner examples.

## Documentation
This will likely not be a strongly documented project as it does not appear to serve much of a purpose right now. If things change in the future, this still might simply serve as some place to start from or reference for us. There is a minor attempt to write some documentation in the code but it is not guaranteed to be easily readable.

## Current Results
At this time, [tch-rs](https://github.com/LaurentMazare/tch-rs) does not appear to support saving the optimizer state which will hurt agent performance if starting from checkpoints often. Additionally, it does not appear to be quite as performant overall as our optimized PyTorch/Python code during training. Because of these limitations, this will likely be put on hold to be a more feature rich reinforcement learning framework.
