#![allow(dead_code)]
use std::sync::Arc;

use crate::{op::Op, storage::Storage, DType, Device};

pub struct Tensor_ {
    storage: Storage,
    shape: Vec<usize>,
    stride: Vec<usize>,
    op: Option<Op>,
}

pub struct Tensor(Arc<Tensor_>);

impl Tensor {
    pub fn zeros(shape: &[usize], dtype: DType, device: Device) -> Self {
        let storage = device.zeros(shape, dtype);
        let tensor_ = Tensor_ {
            storage,
            shape: shape.to_vec(),
            stride: vec![1; shape.len()],
            op: None,
        };
        Tensor(Arc::new(tensor_))
    }

    //The reason for self.0 is Tensor is a tuple struct wapper around Tensor_ with Arc
    //https://doc.rust-lang.org/std/keyword.self.html
    pub fn dtype(&self) -> DType {
        self.0.storage.dtype()
    }

    pub fn device(&self) -> Device {
        self.0.storage.device()
    }

    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    pub fn stride(&self) -> &[usize] {
        &self.0.stride
    }

    //The rank of a tensor is the number of dimensions or axes it has. In other words, it is the length of the shape of the tensor.
    pub fn rank(&self) -> usize {
        self.0.shape.len()
    }

    //Max number of elements in the Tensor
    pub fn elem_count(&self) -> usize {
        self.0.shape.iter().product()
    }
}
