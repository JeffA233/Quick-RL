// use crate::models::model_base::{DiscreteActPPO, CriticPPO};

// pub struct PPOLearner {
//     policy: Box<dyn DiscreteActPPO>, 
//     critic: Box<dyn CriticPPO>, 
//     n_epochs: usize, 
//     batch_size: usize, 
//     minibatch_size: usize, 
//     clip_range: f32, 
//     ent_coef: f32,
// }

// impl PPOLearner {
//     pub fn new(policy: Box<dyn DiscreteActPPO>, critic: Box<dyn CriticPPO>, n_epochs: usize, batch_size: usize, minibatch_size: usize, clip_range: f32, ent_coef: f32) -> Self {
//         Self {
//             policy,
//             critic,
//             n_epochs,
//             batch_size,
//             minibatch_size,
//             clip_range,
//             ent_coef
//         }
//     }
// }