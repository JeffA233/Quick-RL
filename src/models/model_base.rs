use tch::Tensor;

// technically this could just be a Module from tch but maybe in the future we want to add our own trait functions
pub trait Model {
    fn forward(&mut self, input: &Tensor) -> Tensor;
}