#![allow(dead_code)]
use crate::{cpu_backend::CpuStorage, dtype::WithDType, DType, DeviceError, Shape};

pub trait BackendDevice: Sized + std::fmt::Debug + Clone {
    type Storage: BackendStorage;

    fn new(ordinal: usize) -> Result<Self, DeviceError>;
    fn as_str(&self) -> String;
    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage, DeviceError>;
    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage, DeviceError>;
    fn storage_from_slice<T: WithDType>(&self, data: &[T]) -> Result<Self::Storage, DeviceError>;
    fn storage_from_cpu_storage(
        &self,
        cpu_storage: &CpuStorage,
    ) -> Result<Self::Storage, DeviceError>;
}

pub trait BackendStorage {
    type Device: BackendDevice;

    fn dtype(&self) -> DType;
    fn device(&self) -> &Self::Device;
    fn get_cpu_storage(&self) -> CpuStorage;
}
