use crate::{cpu_backend::CpuStorage, DType, Device};

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
}
