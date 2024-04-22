use crate::{DType, Device};

#[allow(dead_code)]
pub enum Storage {
    Cpu {
        dtype: crate::DType,
        buffer: Vec<u8>,
    },
}

impl Storage {
    pub fn device(&self) -> Device {
        match self {
            Self::Cpu { .. } => Device::Cpu,
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::Cpu { dtype, .. } => *dtype,
        }
    }
}
