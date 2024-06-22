use cudarc::driver::DriverError;
use thiserror::Error;

use crate::error::DeviceError;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum CudaStorageError {
    #[error("Custom CpuStorageError {0}")]
    Custom(String),
    #[error("Data Type Mismatch")]
    MismatchDtype,
    #[error("DataType Mismatch in Contiguous Elements")]
    ContiguousElementDtypeMismatch,
    #[error("Empty Array")]
    EmptyArray,
    #[error("Error while Running Op: {0}")]
    OpRunner(String),
    #[error("Device Error in Storage: {}", .source)]
    DeviceError {
        #[from]
        source: DeviceError,
    },
    #[error("Kernel Launch Error: {0}")]
    KernelLaunch(String),
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum CudaDeviceError {
    #[error("Custom CUDA Device Error: {0}")]
    Custom(String),
    #[error("CUDA Driver Device Error {}", .source)]
    DriverDevice {
        #[from]
        source: DriverError,
    },
    #[error("Data Type Mismatch")]
    MismatchDtype,
    #[error("Missing Kernel")]
    MissingKernel(String),
    #[error("Alloc failed {0:?}")]
    AllocFail(Option<String>),
    #[error("Kernel Launch Error: {0}")]
    KernelLaunch(String),
}
