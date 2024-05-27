mod shape;
mod tensor;
use pyo3::prelude::*;

//All Lib Imports
use shape::Shape;
use tensor::Tensor;

/// A Python module implemented in Rust.
#[pymodule]
fn flowtorch_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Shape>()?;
    m.add_class::<Tensor>()?;
    Ok(())
}
