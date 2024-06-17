use crate::{dtype::WithDType, error::Error, Tensor};

pub enum TensorScalar {
    Tensor(Tensor),
    Scalar(Tensor),
}

pub trait TensorOrScalar {
    fn to_tensor_scalar(self) -> Result<TensorScalar, Error>;
}

impl TensorOrScalar for &Tensor {
    fn to_tensor_scalar(self) -> Result<TensorScalar, Error> {
        Ok(TensorScalar::Tensor(self.clone()))
    }
}

impl<T: WithDType> TensorOrScalar for T {
    fn to_tensor_scalar(self) -> Result<TensorScalar, Error> {
        let scalar = Tensor::new(self, &crate::Device::Cpu)?;
        Ok(TensorScalar::Scalar(scalar))
    }
}
