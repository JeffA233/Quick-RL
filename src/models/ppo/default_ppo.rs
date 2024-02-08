use tch::{nn::{self, init, LinearConfig}, Device, Kind, Tensor};
use crate::models::model_base::{Model, DiscreteActPPO, CriticPPO};


pub struct Actor {
    seq: nn::Sequential,
    // actor: Linear,
    device: Device,
    n_act: i64,
    n_in: i64,
}

impl Actor {
    pub fn new(p: &nn::Path, n_act: i64, n_in: i64, mut layers: Vec<i64>, config: Option<LinearConfig>) -> Self {
        assert!(layers.len() > 2, "layer count must be at least 2 for actor");
        // default LinearConfig with kaiming, identical to calling LinearConfig::default() for now
        let lin_conf = config.unwrap_or(LinearConfig { ws_init: init::DEFAULT_KAIMING_NORMAL, bs_init: None, bias: true });
        // define layer functions
        let layer_func = |in_dim: i64, out_dim: i64, layer_str: String| nn::linear(p / layer_str, in_dim, out_dim, lin_conf);
        let activation_func = |xs: &Tensor| xs.relu();
        // start building network
        let mut seq = nn::seq();
        seq = seq.add(layer_func(n_in, layers[0], String::from("al0")));
        seq = seq.add_fn(activation_func);
        for i in 1..layers.len()-1 {
            let layer_str = String::from("al") + &i.to_string();
            seq = seq.add(layer_func(layers[i], layers[i+1], layer_str));
            seq = seq.add_fn(activation_func);
        }
        // let actor = nn::linear(p / "alout", net_dim, n_act, lin_conf);
        seq = seq.add(nn::linear(p / "alout", layers.pop().unwrap(), n_act, lin_conf));
        seq = seq.add_fn(move |xs| xs.softmax(-1, None));
        let device = p.device();

        Self {
            seq,
            // actor,
            device,
            n_act,
            n_in,
        }
    }
}

impl Model for Actor {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // let ten = input.apply(&self.seq);
        // ten.apply(&self.actor)
        input.apply(&self.seq)
    }
}

impl DiscreteActPPO for Actor {
    // fn forward(&mut self, input: &Tensor) -> Tensor {
    //     let ten = input.apply(&self.seq);
    //     ten.apply(&self.actor)
    // }

    fn get_act_prob(&mut self, input: &Tensor, deterministic: bool) -> (Tensor, Tensor) {
        let mut probs = self.forward(input);
        probs = probs.view([-1, self.n_act]);
        probs = probs.clamp(1e-11, 1.);

        if deterministic {
            return (probs.argmax(None, false), Tensor::from_slice(&[0.]))
        }

        let action = probs.multinomial(1, true);
        let log_probs = probs.log().gather(-1, &action, false);

        (action.flatten(0, -1), log_probs.flatten(0, -1))
    }

    fn get_prob_entr(&mut self, input: &Tensor, acts: &Tensor) -> (Tensor, Tensor) {
        let acts = acts.internal_cast_long(true);
        let mut probs = self.forward(input);
        probs = probs.view([-1, self.n_act]);
        probs = probs.clamp(1e-11, 1.);

        let log_probs = probs.log();
        let log_probs_act = log_probs.gather(-1, &acts.unsqueeze(-1), false);
        let entropy = -(log_probs * probs).sum_dim_intlist(-1, false, Kind::Float);

        (log_probs_act.squeeze(), entropy.mean(None))
    }

    fn get_device(&self) -> Device {
        self.device
    }

    fn get_n_in(&self) -> i64 {
        self.n_in
    }

    // fn set_device(&mut self, dev: Device) {
    //     self.device = dev;
    // }
}

pub struct Critic {
    seq: nn::Sequential,
    // critic: Linear,
    device: Device,
    n_in: i64,
}

impl Critic {
    pub fn new(p: &nn::Path, n_in: i64, mut layers: Vec<i64>, config: Option<LinearConfig>) -> Self {
        assert!(layers.len() > 2, "layer count must be at least 2 for critic");
        // let layer_iter = layers.iter();
        // default LinearConfig with kaiming
        let lin_conf = config.unwrap_or(LinearConfig { ws_init: init::DEFAULT_KAIMING_NORMAL, bs_init: None, bias: true });
        // define layer functions
        let layer_func = |in_dim: i64, out_dim: i64, layer_str: String| nn::linear(p / layer_str, in_dim, out_dim, lin_conf);
        let activation_func = |xs: &Tensor| xs.relu();
        // start building network
        let mut seq = nn::seq();
        seq = seq.add(layer_func(n_in, layers[0], String::from("cl0")));
        seq = seq.add_fn(activation_func);
        for i in 1..layers.len()-1 {
            let layer_str = String::from("cl") + &i.to_string();
            seq = seq.add(layer_func(layers[i], layers[i+1], layer_str));
            seq = seq.add_fn(activation_func);
        }
        // let critic = nn::linear(p / "clout", net_dim, 1, lin_conf);
        seq = seq.add(nn::linear(p / "clout", layers.pop().unwrap(), 1, lin_conf));
        let device = p.device();

        Self {
            seq,
            // critic,
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
        // let ten = input.apply(&self.seq);
        // ten.apply(&self.critic)
        input.apply(&self.seq)
    }
}

impl CriticPPO for Critic {
    fn get_device(&self) -> Device {
        self.device
    }

    fn get_n_in(&self) -> i64 {
        self.n_in
    }

    fn set_device(&mut self, dev: Device) {
        self.device = dev;
    }
}

// pub struct PPOPreProcess {

// }

// impl Model for PPOPreProcess {
//     fn forward(&mut self, input: &Tensor) -> Tensor {
//         input.copy()
//     }
// }
