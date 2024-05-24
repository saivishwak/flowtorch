//! Tensors are N-dimensional matrixes of elements using a single data type.
use std::sync::{Arc, RwLock};

use crate::{
    dtype::WithDType,
    layout::{Layout, Stride},
    ndarray::NdArray,
    shape::Shape,
    storage::Storage,
    DType, Device, Error,
};

#[derive(Debug)]
pub struct Tensor_ {
    storage: Arc<RwLock<Storage>>, //Arc ensures that when clone is performed the data is not replicated
    layout: Layout,
    device: Device,
    //op: Option<Op>,
}

#[derive(Debug)]
pub struct Tensor(Arc<Tensor_>);

impl Tensor {
    pub fn new<D>(array: D, device: &Device) -> Result<Self, Error>
    where
        D: NdArray,
    {
        let shape = array.shape()?;
        let storage = device.from_array(array)?;
        return Self::from_storage(storage, shape);
    }

    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self, Error> {
        let shape = shape.into();
        let storage = device.zeros(&shape, dtype)?;
        return Self::from_storage(storage, shape);
    }

    pub fn ones<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self, Error> {
        let shape = shape.into();
        let storage = device.ones(&shape, dtype)?;
        return Self::from_storage(storage, shape);
    }

    pub fn from_vec<S: Into<Shape>, D: WithDType>(
        data: Vec<D>,
        shape: S,
        device: &Device,
    ) -> Result<Self, Error> {
        let shape = shape.into();
        let buffer_size = data.len();
        if buffer_size != shape.elem_count() {
            return Err(Error::TensorInit(
                Some(D::get_dtype()),
                String::from("Provided shape and length of Data does not match"),
            ));
        }
        let storage = device.storage_owned(data)?;
        return Self::from_storage(storage, shape);
    }

    fn from_storage<S: Into<Shape>>(storage: Storage, shape: S) -> Result<Self, Error> {
        let device = storage.device();
        let tensor_ = Tensor_ {
            storage: Arc::new(RwLock::new(storage)),
            layout: Layout::contiguous(shape),
            device,
        };
        Ok(Tensor(Arc::new(tensor_)))
    }

    pub fn reshape<S: Into<Shape>>(&mut self, shape: S) -> Result<Self, Error> {
        let shape: Shape = shape.into();
        if shape.elem_count() != self.elem_count() {
            //Shape mismatch
            return Err(Error::Shape(crate::ShapeError::ReshapeError(String::from(
                "Mismatch in Elements",
            ))));
        }
        //Right now Reshape is only supported for contiguous Tensor.
        // The Arc<RwLock<Storage>> ensures that the clone does not create new data in heap
        if self.0.layout.is_contiguous() {
            let storage = self.0.storage.clone();
            let device = self.get_storage_ref().device();
            let tensor_ = Tensor_ {
                storage,
                layout: Layout::contiguous_with_offset(shape, self.0.layout.offset),
                device,
            };
            return Ok(Tensor(Arc::new(tensor_)));
        }
        return Err(Error::Shape(crate::ShapeError::ReshapeError(String::from(
            "Tensor Layout not contiguous, As of now we only support contiguous memory layout",
        ))));
    }

    /*
    View is same as reshape
     */
    pub fn view<S: Into<Shape>>(&mut self, shape: S) -> Result<Self, Error> {
        return self.reshape(shape);
    }

    pub fn get_storage_ref(&self) -> std::sync::RwLockReadGuard<Storage> {
        let storage = self.0.storage.read().unwrap();
        return storage;
    }

    pub fn dims(&self) -> Vec<usize> {
        self.shape().dims().to_vec()
    }

    pub fn dtype(&self) -> DType {
        self.get_storage_ref().dtype()
    }

    pub fn device(&self) -> Device {
        self.0.device
    }

    pub fn shape(&self) -> Shape {
        self.0.layout.get_shape()
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
