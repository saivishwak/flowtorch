use crate::{
    backend::BackendStorage,
    cpu_backend::CpuStorage,
    error::{Error, StorageError},
    layout::Layout,
    ops::{BinaryOpT, UnaryOpT},
    DType, Device,
};

#[derive(Debug)]
pub enum Storage {
    Cpu(CpuStorage),
    Cuda(crate::cuda::CudaStorage),
}

impl Storage {
    pub fn device(&self) -> Device {
        match self {
            Self::Cpu(_) => Device::Cpu,
            Self::Cuda(storage) => Device::Cuda(storage.device().clone()),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::Cpu(storage) => storage.dtype(),
            Self::Cuda(storage) => storage.dtype(),
        }
    }

    pub fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self, StorageError> {
        match self {
            Self::Cpu(storage) => Ok(Storage::Cpu(storage.to_dtype(layout, dtype)?)),
            Self::Cuda(storage) => Ok(Storage::Cuda(storage.to_dtype(layout, dtype)?)),
        }
    }

    pub fn get_cpu_storage(&self) -> CpuStorage {
        match self {
            Self::Cpu(storage) => storage.get_cpu_storage(),
            Self::Cuda(storage) => storage.get_cpu_storage(),
        }
    }

    pub fn get_storage_ref(&self) -> &Self {
        self
    }

    pub(crate) fn index_select(
        &self,
        other: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        dim: usize,
    ) -> Result<Self, StorageError> {
        match (self, other) {
            (Self::Cpu(lhs), Self::Cpu(rhs)) => {
                let storage = lhs.index_select(rhs, lhs_layout, rhs_layout, dim)?;
                Ok(Storage::Cpu(storage))
            }
            (Self::Cuda(_), Self::Cuda(_)) => todo!(),
            _ => Err(StorageError::Unknown),
        }
    }

    pub(crate) fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self, StorageError> {
        match (self, rhs) {
            (Self::Cpu(lhs), Self::Cpu(rhs)) => {
                let storage = lhs.binary_impl::<B>(rhs, lhs_layout, rhs_layout)?;
                Ok(Storage::Cpu(storage))
            }
            (Storage::Cuda(lhs), Storage::Cuda(rhs)) => {
                let storage = lhs.binary_impl::<B>(rhs, lhs_layout, rhs_layout)?;
                Ok(Storage::Cuda(storage))
            }
            _ => Err(StorageError::DeviceMismatch),
        }
    }

    pub(crate) fn unary_impl<U: UnaryOpT>(&self, layout: &Layout) -> Result<Self, StorageError> {
        match self {
            Self::Cpu(lhs) => {
                let storage = lhs.unary_impl::<U>(layout)?;
                Ok(Storage::Cpu(storage))
            }
            Self::Cuda(lhs) => {
                let storage = lhs.unary_impl::<U>(layout)?;
                Ok(Storage::Cuda(storage))
            }
        }
    }

    pub(crate) fn cmp(
        &self,
        _rhs: &Self,
        _lhs_layout: &Layout,
        _rhs_layout: &Layout,
    ) -> Result<Self, Error> {
        todo!()
    }

    pub(crate) fn equal(&self, other: &Self, self_layout: &Layout, other_layout: &Layout) -> bool {
        match (self, other) {
            (Self::Cpu(lhs), Self::Cpu(rhs)) => lhs.equal(rhs, self_layout, other_layout),
            (Storage::Cuda(lhs), Storage::Cuda(rhs)) => lhs.equal(rhs, self_layout, other_layout),
            _ => false,
        }
    }
}
