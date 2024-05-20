//! Tensors are N-dimensional matrixes of elements using a single data type.
use std::sync::Arc;

use crate::{
    device::NdArray,
    layout::{Layout, Stride},
    shape::Shape,
    storage::Storage,
    DType, Device,
};

#[derive(Debug)]
pub struct Tensor_ {
    storage: Storage,
    layout: Layout,
    device: Device, //op: Option<Op>,
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
        let device = storage.device();
        let tensor_ = Tensor_ {
            storage,
            layout: Layout::contiguous(shape),
            device,
        };
        Ok(Tensor(Arc::new(tensor_)))
    }

    pub fn reshape<S: Into<Shape>>(&mut self, shape: S) -> Result<Self, ()> {
        let shape: Shape = shape.into();
        if shape.elem_count() != self.elem_count() {
            //Shape mismatch
            return Err(());
        }
        if self.0.layout.is_contiguous() {
            let storage = self.0.storage.clone();
            let device = storage.device();
            let tensor_ = Tensor_ {
                storage,
                layout: Layout::contiguous_with_offset(shape, self.0.layout.offset),
                device,
            };
            return Ok(Tensor(Arc::new(tensor_)));
        }
        // Not yet handling the Fotran Contiguous style
        Err(())
    }

    //The reason for self.0 is Tensor is a tuple struct wapper around Tensor_ with Arc
    //https://doc.rust-lang.org/std/keyword.self.html
    pub fn dtype(&self) -> DType {
        self.0.storage.dtype()
    }

    pub fn device(&self) -> Device {
        self.0.device
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.layout.get_shape().dims().clone()
    }

    pub fn stride(&self) -> Stride {
        self.0.layout.get_stride()
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
