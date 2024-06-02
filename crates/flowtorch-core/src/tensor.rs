//! Tensors are N-dimensional matrixes of elements using a single data type.
use std::sync::{Arc, RwLock};

use crate::{
    dtype::WithDType,
    layout::{Layout, Stride},
    ndarray::NdArray,
    shape::Shape,
    storage::Storage,
    DType, Device, Error, ShapeError,
};

#[derive(Debug, Clone)]
pub struct Tensor_ {
    storage: Arc<RwLock<Storage>>, //Arc ensures that when clone is performed the data is not replicated
    layout: Layout,
    device: Device,
    //op: Option<Op>,
}

#[derive(Debug, Clone)]
pub struct Tensor(Arc<Tensor_>);

impl std::ops::Deref for Tensor {
    type Target = Tensor_;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl Tensor {
    /* Creation Ops */

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

    /* Indexing, Slicing Ops */

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
        if self.is_layout_contiguous() {
            let storage = self.get_storage_clone();
            let device = self.get_storage_ref().device();
            let tensor_ = Tensor_ {
                storage,
                layout: Layout::contiguous_with_offset(shape, self.layout.start_offset()),
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

    /**
     * Implementation similar to https://pytorch.org/docs/stable/generated/torch.narrow.html
     */
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self, Error> {
        let shape = self.shape();
        let dims = shape.dims();

        if start > dims[dim] {
            return Err(Error::Shape(ShapeError::Narrow(String::from(
                "Start > Dimension length",
            ))));
        }

        if (start + len) > dims[dim] {
            return Err(Error::Shape(ShapeError::Narrow(String::from(
                "Start + Length > Dimension length",
            ))));
        }

        if start == 0 && dims[dim] == len {
            Ok(self.clone())
        } else {
            let layout = self.layout.narrow(dim, start, len)?;
            let tensor_ = Tensor_ {
                storage: self.storage.clone(),
                layout,
                device: self.device.clone(),
            };
            Ok(Tensor(Arc::new(tensor_)))
        }
    }

    pub fn squeeze(&self, dim: usize) -> Result<Self, Error> {
        let dims = self.dims();
        if dims[dim] == 1 {
            let mut dims = dims.to_vec();
            let mut strides = self.stride().to_vec();
            dims.remove(dim);
            strides.remove(dim);
            let tensor_ = Tensor_ {
                storage: self.storage.clone(),
                layout: Layout::new(dims.into(), strides, self.layout.start_offset()),
                device: self.device.clone(),
            };
            Ok(Tensor(Arc::new(tensor_)))
        } else {
            Ok(self.clone())
        }
    }

    // Two tensors are euqal if their shapes and elements are equal
    pub fn equal(&self, other: &Self) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let self_storage = &*self.get_storage_ref();
        let other_storage = &*other.get_storage_ref();
        let self_offset = (self.layout.start_offset(), self.elem_count());
        let other_offset = (other.layout.start_offset(), other.elem_count());

        //Element wise comparision
        return self_storage.equal(other_storage, self_offset, other_offset);
    }

    /* Access Methods */

    pub fn get_storage_ref(&self) -> std::sync::RwLockReadGuard<Storage> {
        let storage = self.0.storage.read().unwrap(); //Need to do better error handling here, instead of unwrap
        return storage;
    }

    pub(crate) fn get_storage_clone(&self) -> Arc<RwLock<Storage>> {
        self.storage.clone()
    }

    pub fn dims(&self) -> Vec<usize> {
        self.shape().dims().to_vec()
    }

    pub fn dtype(&self) -> DType {
        self.get_storage_ref().dtype()
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn shape(&self) -> Shape {
        self.layout.shape()
    }

    pub fn stride(&self) -> Stride {
        self.layout.stride()
    }

    pub(crate) fn is_layout_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    //The rank of a tensor is the number of dimensions or axes it has. In other words, it is the length of the shape of the tensor.
    pub fn rank(&self) -> usize {
        self.layout.shape().rank()
    }

    //Max number of elements in the Tensor
    pub fn elem_count(&self) -> usize {
        self.layout.shape().elem_count()
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.equal(other)
    }
}
