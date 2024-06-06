use crate::{
    cpu_backend::CpuStorage,
    layout::Layout,
    ops::{BinaryOpT, UnaryOpT},
    DType, Device, Error,
};

pub trait BaseStorage {
    fn cpu_get_raw(&self) -> Box<&CpuStorage>;
}

#[derive(Debug)]
pub enum Storage {
    Cpu(CpuStorage),
}

impl Storage {
    pub fn device(&self) -> Device {
        match self {
            Self::Cpu(_) => Device::Cpu,
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::Cpu(storage) => match storage {
                CpuStorage::U8(_) => DType::U8,
                CpuStorage::U32(_) => DType::U32,
                CpuStorage::I32(_) => DType::I32,
                CpuStorage::I64(_) => DType::I64,
                CpuStorage::F32(_) => DType::F32,
                CpuStorage::F64(_) => DType::F64,
            },
        }
    }

    pub fn cpu_get_raw(&self) -> Box<&CpuStorage> {
        match self {
            Self::Cpu(storage) => {
                return storage.cpu_get_raw();
            }
        }
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
        }
    }

    pub(crate) fn binary_impl<B: BinaryOpT>(&self, rhs: &Self) -> Result<Self, Error> {
        match (self, rhs) {
            (Self::Cpu(lhs), Self::Cpu(rhs)) => {
                let storage = lhs.binary_impl::<B>(rhs)?;
                Ok(Storage::Cpu(storage))
            }
        }
    }

    pub(crate) fn unary_impl<U: UnaryOpT>(&self) -> Result<Self, Error> {
        match self {
            Self::Cpu(lhs) => {
                let storage = lhs.unary_impl::<U>()?;
                Ok(Storage::Cpu(storage))
            }
        }
    }

    pub(crate) fn equal(
        &self,
        other: &Self,
        self_offset: (usize, usize),
        other_offset: (usize, usize),
    ) -> bool {
        match (self, other) {
            (Self::Cpu(lhs), Self::Cpu(rhs)) => lhs.equal(rhs, self_offset, other_offset),
        }
    }
}
