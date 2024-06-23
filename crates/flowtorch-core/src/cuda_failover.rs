#![allow(dead_code, unused_variables)]
#[cfg(not(tarpaulin_include))]
// This file is used as a placeholder non cuda compilation of the project for non CUDA
use crate::backend::{BackendDevice, BackendStorage};
use crate::dtype::DType;
use crate::error::StorageError;
use crate::ops::BinaryOpT;
use crate::ops::UnaryOpT;
use crate::Layout;
use thiserror::Error;

macro_rules! fail {
    () => {
        unimplemented!("Cuda Not Enabled")
    };
}

#[derive(Debug, Clone)]
pub struct CudaDevice {}

impl CudaDevice {
    pub fn new(ordinal: usize) -> Result<Self, crate::error::DeviceError> {
        fail!()
    }
}

impl BackendDevice for CudaDevice {
    type Storage = CudaStorage;

    fn zeros_impl(
        &self,
        shape: &crate::Shape,
        dtype: crate::DType,
    ) -> Result<Self::Storage, crate::error::DeviceError> {
        fail!()
    }

    fn ones_impl(
        &self,
        shape: &crate::Shape,
        dtype: crate::DType,
    ) -> Result<Self::Storage, crate::error::DeviceError> {
        fail!()
    }

    fn as_str(&self) -> String {
        fail!()
    }

    fn new(ordinal: usize) -> Result<Self, crate::error::DeviceError> {
        fail!()
    }

    fn storage_from_slice<T: crate::dtype::WithDType>(
        &self,
        data: &[T],
    ) -> Result<Self::Storage, crate::error::DeviceError> {
        fail!()
    }

    fn storage_from_cpu_storage(
        &self,
        cpu_storage: &crate::cpu_backend::CpuStorage,
    ) -> Result<Self::Storage, crate::error::DeviceError> {
        fail!()
    }
}

#[derive(Debug)]
pub struct CudaStorage {}

impl CudaStorage {}

impl BackendStorage for CudaStorage {
    type Device = CudaDevice;

    fn get_cpu_storage(&self) -> crate::cpu_backend::CpuStorage {
        fail!()
    }

    fn dtype(&self) -> crate::DType {
        fail!()
    }

    fn device(&self) -> &Self::Device {
        fail!()
    }

    fn to_dtype(&self, layout: &crate::layout::Layout, dtype: DType) -> Result<Self, StorageError> {
        fail!()
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self, StorageError> {
        fail!()
    }
    fn unary_impl<U: UnaryOpT>(&self, layout: &Layout) -> Result<Self, StorageError> {
        fail!()
    }
    fn equal(&self, rhs: &Self, self_layout: &Layout, other_layout: &Layout) -> bool {
        fail!()
    }
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum CudaStorageError {
    #[error("Custom CpuStorageError {0}")]
    Custom(String),
    #[error("Data Type Mismatch")]
    MismatchDtype,
    #[error("DataType Mismatch in Contiguous Elements")]
    ContiguousElementDtypeMismatch,
    #[error("Empty Array")]
    EmptyArray,
    #[error("Error while Running Op: {0}")]
    OpRunner(String),
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum CudaDeviceError {
    #[error("Custom CUDA Device Error: {0}")]
    Custom(String),
    #[error("CUDA Driver Device Error")]
    DriverDevice {},
    #[error("Data Type Mismatch")]
    MismatchDtype,
    #[error("Missing Kernel")]
    MissingKernel(String),
    #[error("Alloc failed {0:?}")]
    AllocFail(Option<String>),
}
