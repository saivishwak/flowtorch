#![allow(unused_imports)]
use std::{sync::Arc, vec};

use crate::{
    device::NdArray,
    layout::{Layout, Strides},
    op::Op,
    shape::Shape,
    storage::{self, Storage},
    DType, Device,
};

#[derive(Debug)]
pub struct Tensor_ {
    storage: Storage,
    layout: Layout,
    //op: Option<Op>,
}

#[derive(Debug)]
pub struct Tensor(Arc<Tensor_>);

impl Tensor {
    pub fn new<D>(array: D, device: &Device) -> Result<Self, ()>
    where
        D: NdArray,
    {
        let shape = array.shape()?;
        let storage = device.from_array(array)?;
        return Self::from_storage(storage, shape);
    }

    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self, ()> {
        let shape = shape.into();
        let storage = device.zeros(&shape, dtype)?;
        return Self::from_storage(storage, shape);
    }

    pub fn ones<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self, ()> {
        let shape = shape.into();
        let storage = device.ones(&shape, dtype)?;
        return Self::from_storage(storage, shape);
    }

    fn from_storage<S: Into<Shape>>(storage: Storage, shape: S) -> Result<Self, ()> {
        let tensor_ = Tensor_ {
            storage,
            layout: Layout::contiguous(shape),
        };
        Ok(Tensor(Arc::new(tensor_)))
    }

    //The reason for self.0 is Tensor is a tuple struct wapper around Tensor_ with Arc
    //https://doc.rust-lang.org/std/keyword.self.html
    pub fn dtype(&self) -> DType {
        self.0.storage.dtype()
    }

    pub fn device(&self) -> Device {
        self.0.storage.device()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.layout.get_shape().dims().clone()
    }

    pub fn strides(&self) -> Strides {
        self.0.layout.get_strides()
    }

    //The rank of a tensor is the number of dimensions or axes it has. In other words, it is the length of the shape of the tensor.
    pub fn rank(&self) -> usize {
        self.0.layout.get_shape().rank()
    }

    //Max number of elements in the Tensor
    pub fn elem_count(&self) -> usize {
        self.0.layout.get_shape().elem_count()
    }
}
