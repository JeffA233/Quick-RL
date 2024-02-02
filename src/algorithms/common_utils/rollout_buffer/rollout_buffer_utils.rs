use bytebuffer::ByteBuffer;
use tch::{Tensor, TchError, nn::VarStore};



// const ROLLOUT_KEY: &str = "rollouts";


pub fn tensor_to_bytes(tensor: &Tensor) -> Result<Vec<u8>, TchError> {
    let mut buffer = ByteBuffer::new();

    let op = tensor.save_to_stream(&mut buffer);

    match op {
        Ok(_) => Ok(buffer.into_vec()),
        Err(e) => Err(e)
    }
}

pub fn varstore_to_bytes(vs: &VarStore) -> Result<Vec<u8>, TchError> {
    let mut buffer = ByteBuffer::new();

    let op = vs.save_to_stream(&mut buffer);

    match op {
        Ok(_) => Ok(buffer.into_vec()),
        Err(e) => Err(e)
    }
}