use tch::Tensor;

// technically this could just be a Module from tch but maybe in the future we want to add our own trait functions
pub trait Model {
    fn forward(&mut self, input: &Tensor) -> Tensor;
}

pub trait DiscreteActPPO: Model {
    // fn forward(&mut self, input: &Tensor) -> Tensor;
    fn get_act_prob(&mut self, input: &Tensor, deterministic: bool) -> (Tensor, Tensor);
    fn get_prob_entr(&mut self, input: &Tensor, acts: &Tensor) -> (Tensor, Tensor);
}

pub trait CriticPPO: Model {
    // fn forward(&mut self, input: &Tensor) -> Tensor;
}