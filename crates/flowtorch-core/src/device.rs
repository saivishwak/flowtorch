use crate::ndarray::NdArray;
use crate::DeviceError;
use crate::{cpu_backend::CpuDevice, dtype::WithDType, shape::Shape, storage::Storage, DType};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
}

impl Device {
    pub fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Storage, DeviceError> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice::zeros(shape, dtype);
                if let Ok(s) = storage {
                    return Ok(Storage::Cpu(s));
                } else {
                    return Err(DeviceError::new(crate::DeviceErrorKind::ZerosFail));
                }
            }
        }
    }

    pub fn ones(&self, shape: &Shape, dtype: DType) -> Result<Storage, DeviceError> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice::ones(shape, dtype);
                if let Ok(s) = storage {
                    return Ok(Storage::Cpu(s));
                } else {
                    return Err(DeviceError::new(crate::DeviceErrorKind::OnesFail));
                }
            }
        }
    }

    pub fn storage_owned<S: WithDType>(&self, data: Vec<S>) -> Result<Storage, DeviceError> {
        match self {
            Device::Cpu => {
                let storage = S::to_cpu_storage(&data);
                return Ok(Storage::Cpu(storage));
            }
        }
    }

    pub fn from_array<D: NdArray>(&self, array: D) -> Result<Storage, DeviceError> {
        match self {
            Device::Cpu => {
                let storage = array.to_cpu_storage();
                match storage {
                    Ok(s) => return Ok(Storage::Cpu(s)),
                    Err(_) => {
                        return Err(DeviceError::new(crate::DeviceErrorKind::FromArrayFailure));
                    }
                }
            }
        }
    }
}
