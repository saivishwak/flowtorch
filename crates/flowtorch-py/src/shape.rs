use flowtorch_core::Shape as ExternalShape;
use pyo3::prelude::*;

#[pyclass]
pub struct Shape {
    pub external_shape: ExternalShape,
}

#[pymethods]
impl Shape {
    #[new]
    fn new(dims: Vec<usize>) -> Self {
        Shape {
            external_shape: ExternalShape::new(dims),
        }
    }
}
