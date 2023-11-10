use tch::Tensor;

pub trait Model {
    fn forward(&mut self, input: &Tensor) -> Tensor;
}