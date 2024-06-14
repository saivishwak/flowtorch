#![allow(dead_code, unused_variables)]
// This file is used as a placeholder non cuda compilation of the project
use crate::backend::{BackendDevice, BackendStorage};
use crate::ops::BinaryOpT;
use crate::ops::UnaryOpT;
use crate::Error;

macro_rules! fail {
    () => {
        unimplemented!("Cuda Not Enabled")
    };
}

#[derive(Debug, Clone)]
pub struct CudaDevice {}

impl CudaDevice {
    pub fn new(ordinal: usize) -> Result<Self, crate::DeviceError> {
        fail!()
    }
}

impl BackendDevice for CudaDevice {
    type Storage = CudaStorage;

    fn zeros_impl(
        &self,
        shape: &crate::Shape,
        dtype: crate::DType,
    ) -> Result<Self::Storage, crate::DeviceError> {
        fail!()
    }

    fn ones_impl(
        &self,
        shape: &crate::Shape,
        dtype: crate::DType,
    ) -> Result<Self::Storage, crate::DeviceError> {
        fail!()
    }

    fn as_str(&self) -> String {
        fail!()
    }

    fn new(ordinal: usize) -> Result<Self, crate::DeviceError> {
        fail!()
    }

    fn storage_from_slice<T: crate::dtype::WithDType>(
        &self,
        data: &[T],
    ) -> Result<Self::Storage, crate::DeviceError> {
        fail!()
    }

    fn storage_from_cpu_storage(
        &self,
        cpu_storage: &crate::cpu_backend::CpuStorage,
    ) -> Result<Self::Storage, crate::DeviceError> {
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
    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self) -> Result<Self, Error> {
        fail!()
    }
    fn unary_impl<U: UnaryOpT>(&self) -> Result<Self, Error> {
        fail!()
    }
    fn equal(&self, rhs: &Self, self_offset: (usize, usize), other_offset: (usize, usize)) -> bool {
        fail!()
    }
}
