//! Tensors are N-dimensional matrixes of elements using a single data type.
use std::sync::{Arc, RwLock};

use crate::{
    dtype::WithDType,
    layout::{Layout, Stride},
    ndarray::NdArray,
    ops::Op,
    shape::Shape,
    storage::Storage,
    DType, Device, Error, ShapeError,
};

#[derive(Debug, Clone)]
pub struct Tensor_ {
    storage: Arc<RwLock<Storage>>, //Arc ensures that when clone is performed the data is not replicated
    layout: Layout,
    device: Device,
    #[allow(dead_code)]
    op: Option<Op>,
}

/// The Tensor struct
///
/// Tensor is N Dimensional Array with same element type
///
/// # Examples
///
/// ```rust
/// use flowtorch_core::{DType, Device, Error, Tensor};
///
/// fn main() -> Result<(), Error> {
///     let x = Tensor::new(&[1.0f64, 2.0, 3.0], &Device::Cpu)?;
///     let y = Tensor::new(vec![1.0f64, 2.0, 3.0], &Device::Cpu)?;
///     let z = x + y;
///     Ok(())
/// }
/// ```
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

    /// Creates a new Tensor from given NDArray
    ///
    /// The NDArray is a custom n-dimensional array which is converted from Vec and Slices. The New Tensor
    /// will store data in the specified device.
    ///
    /// # Errors
    /// If provided device storage allocation fails, Error is returned.
    /// If the Shape cannot be infered from the Array then Error is returned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flowtorch_core::{DType, Device, Error, Tensor};
    ///
    /// fn main() -> Result<(), Error> {
    ///     let x = Tensor::new(&[1.0f64, 2.0, 3.0], &Device::Cpu)?;
    ///     let y = Tensor::new(vec![1.0f64, 2.0, 3.0], &Device::Cpu)?;
    ///     Ok(())
    /// }
    /// ```
    pub fn new<D>(array: D, device: &Device) -> Result<Self, Error>
    where
        D: NdArray,
    {
        let shape = array.shape()?;
        let storage = device.from_array(array)?;
        return Self::from_storage(storage, shape, None);
    }

    /// Creates a new Tensor with all zeros with specified Shape and Dtype
    ///
    /// # Errors
    /// If provided device storage allocation fails, Error is returned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flowtorch_core::{DType, Device, Error, Tensor};
    ///
    /// fn main() -> Result<(), Error> {
    ///     let x = Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?;
    ///     println!("{}", x);
    ///     Ok(())
    /// }
    /// ```
    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self, Error> {
        let shape = shape.into();
        let storage = device.zeros(&shape, dtype)?;
        return Self::from_storage(storage, shape, None);
    }

    /// Creates a new Tensor with all ones with specified Shape and Dtype
    ///
    /// # Errors
    /// If provided device storage allocation fails, Error is returned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flowtorch_core::{DType, Device, Error, Tensor};
    ///
    /// fn main() -> Result<(), Error> {
    ///     let x = Tensor::ones((2, 2), DType::F32, &Device::Cpu)?;
    ///     println!("{}", x);
    ///     Ok(())
    /// }
    /// ```
    pub fn ones<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self, Error> {
        let shape = shape.into();
        let storage = device.ones(&shape, dtype)?;
        return Self::from_storage(storage, shape, None);
    }

    /// Creates a new Tensor from a Vec
    ///
    /// # Errors
    /// If the shape does not match with the Vec len then an Error is returned.
    /// If provided device storage allocation fails, Error is returned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flowtorch_core::{DType, Device, Error, Tensor};
    ///
    /// fn main() -> Result<(), Error> {
    ///     let x = Tensor::from_vec(vec![1, 2], (2, 1), &Device::Cpu)?;
    ///     println!("{}", x);
    ///     Ok(())
    /// }
    /// ```
    pub fn from_vec<S: Into<Shape>, D: WithDType>(
        data: Vec<D>,
        shape: S,
        device: &Device,
    ) -> Result<Self, Error> {
        let shape = shape.into();
        let buffer_size = data.len();
        if buffer_size != shape.elem_count() {
            return Err(Error::TensorInit(
                Some(D::dtype()),
                String::from("Provided shape and length of Data does not match"),
            ));
        }
        let storage = device.storage_owned(data)?;
        return Self::from_storage(storage, shape, None);
    }

    fn from_storage<S: Into<Shape>>(
        storage: Storage,
        shape: S,
        op: Option<Op>,
    ) -> Result<Self, Error> {
        let device = storage.device();
        let tensor_ = Tensor_ {
            storage: Arc::new(RwLock::new(storage)),
            layout: Layout::contiguous(shape),
            device,
            op,
        };
        Ok(Tensor(Arc::new(tensor_)))
    }

    /* Indexing, Slicing Ops */

    /// Returns a tensor with the same data and number of elements as input, but with the specified shape.
    ///
    /// This does not clone the underlying data but referes it.
    ///
    /// # Errors
    /// If the shape does not match with existing shape, i.e the number of elements should be same.
    /// If the Memory layout is non-contiguous (Fotran). Not supported as of today.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flowtorch_core::{DType, Device, Error, Tensor};
    ///
    /// fn main() -> Result<(), Error> {
    ///     let x = Tensor::from_vec(vec![1, 2], (2, 1), &Device::Cpu)?.reshape((1, 2))?;
    ///     println!("{}", x);
    ///     Ok(())
    /// }
    /// ```
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
            let storage = self.storage.clone();
            let device = self.storage().device();
            let tensor_ = Tensor_ {
                storage,
                layout: Layout::contiguous_with_offset(shape, self.layout.start_offset()),
                device,
                op: None,
            };
            return Ok(Tensor(Arc::new(tensor_)));
        }
        return Err(Error::Shape(crate::ShapeError::ReshapeError(String::from(
            "Tensor Layout not contiguous, As of now we only support contiguous memory layout",
        ))));
    }

    /// View is same as Reshape
    pub fn view<S: Into<Shape>>(&mut self, shape: S) -> Result<Self, Error> {
        return self.reshape(shape);
    }

    /// Returns a new tensor that is a narrowed version of input tensor. \
    /// The dimension dim is input from start to start + length. The returned tensor and input tensor share the same underlying storage.
    /// Implementation similar to https://pytorch.org/docs/stable/generated/torch.narrow.html
    ///
    /// This does not clone the underlying data but referes it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flowtorch_core::{DType, Device, Error, Tensor};
    ///
    /// fn main() -> Result<(), Error> {
    ///     let x = Tensor::new(&[[1], [2]], &Device::Cpu)?.narrow(1, 0, 1)?;
    ///     println!("{}", x);
    ///     Ok(())
    /// }
    /// ```
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
                op: None,
            };
            Ok(Tensor(Arc::new(tensor_)))
        }
    }

    /// Returns a tensor with all specified dimensions of input of size 1 removed.
    ///
    /// This does not clone the underlying data but referes it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flowtorch_core::{DType, Device, Error, Tensor};
    ///
    /// fn main() -> Result<(), Error> {
    ///     let x = Tensor::new(&[[1], [2]], &Device::Cpu)?.squeeze(1)?;
    ///     println!("{}", x);
    ///     Ok(())
    /// }
    /// ```
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
                op: None,
            };
            Ok(Tensor(Arc::new(tensor_)))
        } else {
            Ok(self.clone())
        }
    }

    /// TODO - Need to implement
    pub fn index_select(&self, dim: usize, indexes: &Self) -> Result<Self, Error> {
        if indexes.rank() != 1 {
            return Err(Error::Unknown);
        }

        let indexes_len = indexes.dims().len();
        let mut dims = self.dims().to_vec();
        dims[dim] = indexes_len;

        let storage =
            self.storage()
                .index_select(&indexes.storage(), &self.layout, &indexes.layout, dim)?;
        return Self::from_storage(storage, Shape::from(dims), None);
    }

    /* Binary Ops */

    /// Return true if tensors are equal
    /// Two Tensors are equal if their shapes and elements are equal
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flowtorch_core::{DType, Device, Error, Tensor};
    ///
    /// fn main() -> Result<(), Error> {
    ///     let x = Tensor::new(&[1, 2], &Device::Cpu)?;
    ///     let y = Tensor::new(&[[1], [2]], &Device::Cpu)?.squeeze(1)?;
    ///     println!("{}", x.equal(&y));
    ///     Ok(())
    /// }
    /// ```
    pub fn equal(&self, other: &Self) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let self_storage = &*self.storage();
        let other_storage = &*other.storage();
        let self_offset = (self.layout.start_offset(), self.elem_count());
        let other_offset = (other.layout.start_offset(), other.elem_count());

        //Element wise comparision
        return self_storage.equal(other_storage, self_offset, other_offset);
    }

    pub fn add_(&self, rhs: &Self) -> Result<Self, Error> {
        if self.shape() != rhs.shape() {
            return Err(Error::Unknown);
        }
        let lhs_storage = &*self.storage();
        let rhs_storage = &*rhs.storage();
        let new_storage = lhs_storage.add(rhs_storage)?;
        return Self::from_storage(
            new_storage,
            self.shape().clone(),
            Some(Op::Add(self.clone(), rhs.clone())),
        );
    }

    pub fn mul_(&self, rhs: &Self) -> Result<Self, Error> {
        if self.shape() != rhs.shape() {
            return Err(Error::Unknown);
        }
        let lhs_storage = &*self.storage();
        let rhs_storage = &*rhs.storage();
        let new_storage = lhs_storage.mul(rhs_storage)?;
        return Self::from_storage(
            new_storage,
            self.shape().clone(),
            Some(Op::Mul(self.clone(), rhs.clone())),
        );
    }

    /* Access Methods */

    /// Returns a ReadLock Sotrage of the Tensor
    pub fn storage(&self) -> std::sync::RwLockReadGuard<'_, Storage> {
        self.storage.read().unwrap()
    }

    /// Returns the Dims (Shape as Vec) of the Tensor
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flowtorch_core::{DType, Device, Error, Tensor};
    ///
    /// fn main() -> Result<(), Error> {
    ///     let x = Tensor::new(&[1, 2], &Device::Cpu)?;
    ///     println!("{:?}", x.dims());
    ///     Ok(())
    /// }
    /// ```
    pub fn dims(&self) -> Vec<usize> {
        self.shape().dims().to_vec()
    }

    /// Returns the Dtype of the Tensor elements
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flowtorch_core::{DType, Device, Error, Tensor};
    ///
    /// fn main() -> Result<(), Error> {
    ///     let x = Tensor::new(&[1, 2], &Device::Cpu)?;
    ///     println!("{}", x.dtype());
    ///     Ok(())
    /// }
    /// ```
    pub fn dtype(&self) -> DType {
        self.storage().dtype()
    }

    /// Returns the Device of the Tensor
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flowtorch_core::{DType, Device, Error, Tensor};
    ///
    /// fn main() -> Result<(), Error> {
    ///     let x = Tensor::new(&[1, 2], &Device::Cpu)?;
    ///     println!("{}", x.device());
    ///     Ok(())
    /// }
    /// ```
    pub fn device(&self) -> Device {
        self.device
    }

    /// Returns Shape Struct of the Tensor
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flowtorch_core::{DType, Device, Error, Tensor};
    ///
    /// fn main() -> Result<(), Error> {
    ///     let x = Tensor::new(&[1, 2], &Device::Cpu)?;
    ///     println!("{:?}", x.shape());
    ///     Ok(())
    /// }
    /// ```
    pub fn shape(&self) -> Shape {
        self.layout.shape()
    }

    /// Returns Stride of the Tensor
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flowtorch_core::{DType, Device, Error, Tensor};
    ///
    /// fn main() -> Result<(), Error> {
    ///     let x = Tensor::new(&[1, 2], &Device::Cpu)?;
    ///     println!("{:?}", x.stride());
    ///     Ok(())
    /// }
    /// ```
    pub fn stride(&self) -> Stride {
        self.layout.stride()
    }

    /// Return true if the Tensor memeory layout is Contiguous of Row Major (aka C Type)
    pub(crate) fn is_layout_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Returns the Layout Reference of the Tensor
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The rank of a tensor is the number of dimensions or axes it has. In other words, it is the length of the shape of the tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flowtorch_core::{DType, Device, Error, Tensor};
    ///
    /// fn main() -> Result<(), Error> {
    ///     let x = Tensor::new(&[1, 2], &Device::Cpu)?;
    ///     println!("{}", x.rank());
    ///     Ok(())
    /// }
    /// ```
    pub fn rank(&self) -> usize {
        self.layout.shape().rank()
    }

    /// Max number of elements in the Tensor
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flowtorch_core::{DType, Device, Error, Tensor};
    ///
    /// fn main() -> Result<(), Error> {
    ///     let x = Tensor::new(&[1, 2], &Device::Cpu)?;
    ///     println!("{}", x.elem_count());
    ///     Ok(())
    /// }
    /// ```
    pub fn elem_count(&self) -> usize {
        self.layout.shape().elem_count()
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.equal(other)
    }
}
