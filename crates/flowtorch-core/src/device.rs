use std::fmt::Display;

use crate::array::Array;
use crate::backend::BackendDevice;
use crate::{cpu_backend::CpuDevice, dtype::WithDType, shape::Shape, storage::Storage, DType};
use crate::{DeviceError, Error};

pub enum DeviceT {
    Cpu,
    Cuda(usize),
}

#[derive(Debug, Clone)]
pub enum Device {
    Cpu,
    Cuda(crate::cuda::CudaDevice),
}

impl Device {
    pub fn new(device: DeviceT) -> Result<Self, Error> {
        match device {
            DeviceT::Cpu => Ok(Self::Cpu),
            DeviceT::Cuda(ordinal) => {
                let dev = crate::cuda::CudaDevice::new(ordinal);
                match dev {
                    Ok(device) => Ok(Self::Cuda(device)),
                    Err(_e) => Err(Error::Unknown),
                }
            }
        }
    }

    pub fn is_same_device(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Device::Cpu, Device::Cpu) => true,
            (Device::Cpu, Device::Cuda(_)) => false,
            (Device::Cuda(_), Device::Cpu) => false,
            (Device::Cuda(_), Device::Cuda(_)) => true,
        }
    }

    pub fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Storage, DeviceError> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.zeros_impl(shape, dtype)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(dev) => {
                let storage = dev.zeros_impl(shape, dtype)?;
                Ok(Storage::Cuda(storage))
            }
        }
    }

    pub fn ones(&self, shape: &Shape, dtype: DType) -> Result<Storage, DeviceError> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.ones_impl(shape, dtype)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(dev) => {
                let storage = dev.ones_impl(shape, dtype)?;
                Ok(Storage::Cuda(storage))
            }
        }
    }

    pub fn storage_owned<S: WithDType>(&self, data: Vec<S>) -> Result<Storage, DeviceError> {
        match self {
            Device::Cpu => {
                let storage = S::to_cpu_storage(&data);
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                let cpu_storage = S::to_cpu_storage(&data);
                let storage = device.storage_from_cpu_storage(&cpu_storage)?;
                Ok(Storage::Cuda(storage))
            }
        }
    }

    pub fn from_array<D: Array>(&self, array: D) -> Result<Storage, DeviceError> {
        match self {
            Device::Cpu => {
                let storage = array.to_cpu_storage();
                match storage {
                    Ok(s) => Ok(Storage::Cpu(s)),
                    Err(_) => Err(DeviceError::new(crate::DeviceErrorKind::FromArrayFailure)),
                }
            }
            Device::Cuda(device) => {
                let storage = array.to_cpu_storage();
                match storage {
                    Ok(s) => {
                        let cuda_storage = device.storage_from_cpu_storage(&s)?;
                        Ok(Storage::Cuda(cuda_storage))
                    }
                    Err(_) => Err(DeviceError::new(crate::DeviceErrorKind::FromArrayFailure)),
                }
            }
        }
    }
}

impl Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "Cpu"),
            Self::Cuda(cuda_device) => write!(f, "{}", cuda_device.as_str()),
        }
    }
}
