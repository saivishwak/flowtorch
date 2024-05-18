use crate::{cpu_backend::CpuDevice, shape::Shape, storage::Storage, DType};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
}

impl Device {
    pub fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Storage, ()> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice::zeros(shape, dtype)?;
                return Ok(Storage::Cpu(storage));
            }
        }
    }

    pub fn ones(&self, shape: &Shape, dtype: DType) -> Result<Storage, ()> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice::ones(shape, dtype)?;
                return Ok(Storage::Cpu(storage));
            }
        }
    }
}
