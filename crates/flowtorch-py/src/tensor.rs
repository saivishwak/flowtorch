use flowtorch::Tensor as ExternalTensor;
use pyo3::prelude::*;

#[pyclass]
pub struct Tensor {
    pub external_tensor: ExternalTensor,
}
