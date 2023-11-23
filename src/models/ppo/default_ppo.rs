use tch::{Tensor, nn::{self, init, LinearConfig}, nn::Linear, Device};
use crate::models::model_base::Model;


pub struct Actor {
    seq: nn::Sequential,
    actor: Linear,
    device: Device,
    n_act: i64,
    n_in: i64,
}

impl Actor {
    pub fn new(p: &nn::Path, n_act: i64, n_in: i64, n_layers: usize, net_dim: i64, config: Option<LinearConfig>) -> Self {
        // default LinearConfig with kaiming
        let lin_conf = config.unwrap_or(LinearConfig { ws_init: init::DEFAULT_KAIMING_NORMAL, bs_init: None, bias: true });
        // define layer functions
        let layer_func = |in_dim: i64, out_dim: i64, layer_str: String| nn::linear(p / layer_str, in_dim, out_dim, lin_conf);
        let activation_func = |xs: &Tensor| xs.relu();
        // start building network
        let mut seq = nn::seq();
        seq = seq.add(layer_func(n_in, net_dim, String::from("al0")));
        seq = seq.add_fn(move |xs| activation_func(xs));
        for i in 1..n_layers {
            let layer_str = String::from("al") + &i.to_string();
            seq = seq.add(layer_func(net_dim, net_dim, layer_str));
            seq = seq.add_fn(move |xs| activation_func(xs));
        }
        let actor = nn::linear(p / "alout", 256, n_act, lin_conf);
        let device = p.device();

        Self {
            seq,
            actor,
            device,
            n_act,
            n_in,
        }
    }
}

impl Model for Actor {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let ten = input.apply(&self.seq);
        ten.apply(&self.actor)
    }
}

pub struct Critic {
    seq: nn::Sequential,
    critic: Linear,
    device: Device,
    n_in: i64,
}

impl Critic {
    pub fn new(p: &nn::Path, n_in: i64, n_layers: usize, net_dim: i64, config: Option<LinearConfig>) -> Self {
        // default LinearConfig with kaiming
        let lin_conf = config.unwrap_or(LinearConfig { ws_init: init::DEFAULT_KAIMING_NORMAL, bs_init: None, bias: true });
        // define layer functions
        let layer_func = |in_dim: i64, out_dim: i64, layer_str: String| nn::linear(p / layer_str, in_dim, out_dim, lin_conf);
        let activation_func = |xs: &Tensor| xs.relu();
        // start building network
        let mut seq = nn::seq();
        seq = seq.add(layer_func(n_in, net_dim, String::from("cl0")));
        seq = seq.add_fn(move |xs| activation_func(xs));
        for i in 1..n_layers {
            let layer_str = String::from("cl") + &i.to_string();
            seq = seq.add(layer_func(net_dim, net_dim, layer_str));
            seq = seq.add_fn(move |xs| activation_func(xs));
        }
        let critic = nn::linear(p / "clout", 256, 1, lin_conf);
        let device = p.device();

        Self {
            seq,
            critic,
            device,
            n_in,
        }
    }

    // NOTE: you can't init a sequential from tch so you should just init in the beginning
    // pub fn init(&mut self, init_func: nn::Init) {
    //     self.critic.ws.init(init_func);
    //     if let Some(bs) = &mut self.critic.bs {*bs = bs.fill_(0.);}
    // }
}

impl Model for Critic {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let ten = input.apply(&self.seq);
        ten.apply(&self.critic)
    }
}

// pub struct PPOPreProcess {

// }

// impl Model for PPOPreProcess {
//     fn forward(&mut self, input: &Tensor) -> Tensor {
//         input.copy()
//     }
// }
