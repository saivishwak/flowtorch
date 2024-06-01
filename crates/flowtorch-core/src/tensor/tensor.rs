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

use super::utils;

#[derive(Debug)]
pub(crate) struct Tensor_ {
    pub(crate) storage: Arc<RwLock<Storage>>, //Arc ensures that when clone is performed the data is not replicated
    pub(crate) layout: Layout,
    pub(crate) device: Device,
    //op: Option<Op>,
}

#[derive(Debug)]
pub struct Tensor(pub(crate) Arc<Tensor_>);

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
    view is same as reshape
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

    pub fn as_string(&self, truncate: Option<bool>) -> Result<String, String> {
        let dims = self.dims();
        let strides = self.stride();

        let storage = self.get_storage_ref();

        let binding = storage.cpu_get_raw();
        let storage_data = binding.as_ref();
        let initial_offset = self.0.layout.offset;

        if !self.0.layout.is_contiguous() {
            return Err(String::from("Non Contigous layout not supported"));
        }
        let formatted_string = match storage_data {
            crate::cpu_backend::CpuStorage::U8(data) => {
                utils::as_string(&data, dims, strides, truncate, initial_offset)
            }
            crate::cpu_backend::CpuStorage::U32(data) => {
                utils::as_string(&data, dims, strides, truncate, initial_offset)
            }
            crate::cpu_backend::CpuStorage::I64(data) => {
                utils::as_string(&data, dims, strides, truncate, initial_offset)
            }
            crate::cpu_backend::CpuStorage::F32(data) => {
                utils::as_string(&data, dims, strides, truncate, initial_offset)
            }
            crate::cpu_backend::CpuStorage::F64(data) => {
                utils::as_string(&data, dims, strides, truncate, initial_offset)
            }
        };
        Ok(formatted_string)
    }
}

// Two tensors are euqal if their shapes are equal
impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let self_storage = &*self.get_storage_ref();
        let other_storage = &*other.get_storage_ref();
        let self_offset = (self.0.layout.offset, self.elem_count());
        let other_offset = (other.0.layout.offset, other.elem_count());

        //Element wise comparision
        return self_storage.equal(other_storage, self_offset, other_offset);
    }
}
