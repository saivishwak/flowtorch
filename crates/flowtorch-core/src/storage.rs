use std::usize;

use crate::{cpu_backend::CpuStorage, DType, Device};

pub trait BaseStorage {
    fn cpu_get_raw(&self) -> Box<&CpuStorage>;
}

#[derive(Debug, Clone)]
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

    pub fn equal(
        &self,
        other: &Self,
        self_offset: (usize, usize),
        other_offset: (usize, usize),
    ) -> bool {
        match (self, other) {
            (Self::Cpu(self_data), Self::Cpu(other_data)) => {
                self_data.equal(other_data, self_offset, other_offset)
            }
        }
    }
}
