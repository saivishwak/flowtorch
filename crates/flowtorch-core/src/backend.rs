use crate::{
    cpu_backend::CpuStorage,
    dtype::WithDType,
    layout::Layout,
    ops::{BinaryOpT, UnaryOpT},
    DType, DeviceError, Error, Shape,
};

pub trait BackendDevice: Sized + std::fmt::Debug + Clone {
    type Storage: BackendStorage;

    #[allow(dead_code)]
    fn new(ordinal: usize) -> Result<Self, DeviceError>;
    fn as_str(&self) -> String;
    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage, DeviceError>;
    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage, DeviceError>;
    #[allow(dead_code)]
    fn storage_from_slice<T: WithDType>(&self, data: &[T]) -> Result<Self::Storage, DeviceError>;
    fn storage_from_cpu_storage(
        &self,
        cpu_storage: &CpuStorage,
    ) -> Result<Self::Storage, DeviceError>;
}

pub trait BackendStorage: Sized + std::fmt::Debug {
    type Device: BackendDevice;

    fn dtype(&self) -> DType;
    fn device(&self) -> &Self::Device;

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self, Error>;

    fn get_cpu_storage(&self) -> CpuStorage;

    fn unary_impl<U: UnaryOpT>(&self) -> Result<Self, Error>;
    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self) -> Result<Self, Error>;
    fn equal(&self, rhs: &Self, self_offset: (usize, usize), other_offset: (usize, usize)) -> bool;
}
