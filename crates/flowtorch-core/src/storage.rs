use crate::{
    backend::BackendStorage,
    cpu_backend::CpuStorage,
    layout::Layout,
    ops::{BinaryOpT, UnaryOpT},
    DType, Device, Error,
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

    pub fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self, Error> {
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
    ) -> Result<Self, Error> {
        match (self, other) {
            (Self::Cpu(lhs), Self::Cpu(rhs)) => {
                let storage = lhs.index_select(rhs, lhs_layout, rhs_layout, dim)?;
                Ok(Storage::Cpu(storage))
            }
            (Self::Cuda(_), Self::Cuda(_)) => todo!(),
            _ => Err(Error::Unknown),
        }
    }

    pub(crate) fn binary_impl<B: BinaryOpT>(&self, rhs: &Self) -> Result<Self, Error> {
        match (self, rhs) {
            (Self::Cpu(lhs), Self::Cpu(rhs)) => {
                let storage = lhs.binary_impl::<B>(rhs)?;
                Ok(Storage::Cpu(storage))
            }
            (Storage::Cuda(lhs), Storage::Cuda(rhs)) => {
                let storage = lhs.binary_impl::<B>(rhs)?;
                Ok(Storage::Cuda(storage))
            }
            _ => Err(Error::Unknown),
        }
    }

    pub(crate) fn unary_impl<U: UnaryOpT>(&self) -> Result<Self, Error> {
        match self {
            Self::Cpu(lhs) => {
                let storage = lhs.unary_impl::<U>()?;
                Ok(Storage::Cpu(storage))
            }
            Storage::Cuda(lhs) => {
                let storage = lhs.unary_impl::<U>()?;
                Ok(Storage::Cuda(storage))
            }
        }
    }

    pub(crate) fn cmp(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self, Error> {
        todo!()
    }

    pub(crate) fn equal(
        &self,
        other: &Self,
        self_offset: (usize, usize),
        other_offset: (usize, usize),
    ) -> bool {
        match (self, other) {
            (Self::Cpu(lhs), Self::Cpu(rhs)) => lhs.equal(rhs, self_offset, other_offset),
            (Storage::Cuda(lhs), Storage::Cuda(rhs)) => lhs.equal(rhs, self_offset, other_offset),
            _ => false,
        }
    }
}
